function [u_R,w,x_R,f,n]=demo_DDP_CP_STS_ROBOT(model, model_R, model_H, assistance,x0_R)


%Data from the human assistance prediction
x_H = assistance.x_H;
u_H = assistance.u_H;
F = assistance.F;
x0_H=x_H(:,1);



% Weight of input commands
r_vec = blkdiag(eye(model_R.nbVarX)*1,1e5,1e5)*model.wr;
R = sparse(kron(eye(model.nbData-1),r_vec));

% Concatenating all joint limits
T_thets = kron(eye(model.nbData),[inv(tril(ones(model_R.nbVarX,model_R.nbVarX))) zeros(model_R.nbVarX,model_R.nbVarX)]);
X_min_A = kron(ones(model.nbData,1),model_R.xmin);
X_max_A = kron(ones(model.nbData,1),model_R.xmax);


%The human's hand trajectory to follow
for t=1:model.nbData
    xd(:,t) = (model_H.L*[sin(x_H(1:model_H.nbVarX,t)) -cos(x_H(1:model_H.nbVarX,t))])';
end
xd=xd(:);



% Defining important Ids to consider in calculating the gradinet. To speed
% up the code
idc_R = model.nbData-2:model.nbData; %Points to consider the robot's stability
idc_H = model.nbData-2:model.nbData; %Points to consider the human's stability
idq0 = [repmat(1:model_H.nbVarX,model.nbData,1) + ([0:model.nbData-1]*2*model_H.nbVarX)']';
idq =  idq0(:); %Features to follow (only joint angles)


% Transfer matrix of control primitives
nbFct = model_R.nbFct;
t = linspace(0, 1, model.nbData-1);
tMu = linspace(t(1), t(end), nbFct);
phi = zeros(model.nbData-1, nbFct);
for i=1:nbFct
	phi(:,i) = gaussPDF(t, tMu(i), 2E-2);
% 	phi(:,i) = mvnpdf(t', tMu(i), 2E-2);
end
phi = phi ./ repmat(sum(phi,2),1,nbFct); %Optional rescaling
Psi = kron(phi, eye((model_R.nbVarU+2))); 




% Transfer matrix of movement primitives
tx = linspace(0, 1, model.nbData);
tMux = linspace(tx(1), tx(end), nbFct);
phix = zeros(model.nbData, nbFct);
for i=1:nbFct
    phix(:,i) = gaussPDF(tx, tMux(i), 2E-2);
% 	phi(:,i) = mvnpdf(t', tMu(i), 2E-2);
end
phix = phix ./ repmat(sum(phix,2),1,nbFct); %Optional rescaling
Psix = kron(phix, eye((model_R.nbVarX))); 



%% warm starting and imitation trajectory (using k-NN with k=1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Read demonstrations
demos = load('./STS_demonstrations.mat').sts_demo;
demo_input=[];
demo_wuR = [];
demo_wxR = [];
for k = 1 : model.nbDemos
    demo_input = [demo_input;demos{k}.model_H.r_vec(1:3)*1e5 demos{k}.model_H.H demos{k}.model_H.M];
    demo_wuR = [demo_wuR;demos{k}.wu_R'];
    demo_wxR = [demo_wxR;demos{k}.wx_R'];
end

% Normalize the data
demo_input(:,1:3) = log10(demo_input(:,1:3));
Mean_demo = mean(demo_input);
std_demo = std(demo_input);
current_input = [log10(r_vec(1:3)*1e5) model_H.H model_H.M];
concat_data = [current_input;demo_input];
norm_data = (concat_data-Mean_demo)./std_demo;
% Find the nearest neighbor
distances = dist(norm_data);
[~, NN_idx] = min(distances(2:end,1));
w0x = demo_wxR(NN_idx,:)';
w0u = demo_wuR(NN_idx,:)';

% Keeping only the movement primitives of the joint angles (w0x includes movement primitives of the joint angles and velocities)
wx = reshape(w0x,2*model_R.nbVarX,[]);
wx(model_R.nbVarX+1:end,:)=[];

% Extracting the desired imitation trajectory
pd = Psix*wx(:);



%% iLQR
u_zero = [zeros(model_R.nbVarU,model.nbData-1);F]; %Regulate the input commands around this point
w = w0u;
u_R  = Psi*w;
stuck = 0;


for n=1:model.nbIter
	%System evolution
	[x_R, x_H, xc_R,xc_H, Su0_R, Su0_H, J_Rq, J_Hq, J_Rdq, J_Hdq,Jc00_R, Jc00_H] = fwdDyn(x0_R, x0_H, u_R, u_H, model, model_R, model_H);
    Q_R = x_R(:);
    p= Q_R(idq,1);%imitation feature at the current iteration
    X_R = fkin(x_R, model_R,model.nbData);
    X_H = fkin(x_H, model_H,model.nbData);
    [fc0_R, Jc0_R] = CoMcheck(xc_R, Jc00_R, model_R);
    [fc0_H, Jc0_H] = CoMcheck(xc_H, Jc00_H, model_H);
    fc_R = fc0_R(idc_R);
    Jc_R = Jc0_R(idc_R,:);
    fc_H = fc0_H(idc_H);
    Jc_H = Jc0_H(idc_H,:);
    Suq = Su0_R(idq,:);
    % Joint limits deviations (if the system was in the feasible region, the value of proceeding formulations would be zero)
    fmin_R = -(abs(T_thets*x_R(:)-X_min_A) - (T_thets*x_R(:)-X_min_A))/2;
    fmax_R = (abs(T_thets*x_R(:) - X_max_A)+(T_thets*x_R(:) - X_max_A))/2;
    Tmax = diag(sign(fmax_R))*T_thets;
    Tmin = -diag(sign(fmin_R)) * T_thets;
    %Deviation from the human's hand trajectory
	f = X_R(:) - xd;
     % Update value of the coeficients of the CPs!!!
    dw = (Psi'*Suq'*Suq *Psi*model.wi_R+Psi'*Su0_R'*J_Rq'*J_Rq * Su0_R *Psi*model.track +Psi'*(J_Rq*Su0_R-J_Hq*Su0_H)'*(J_Rq*Su0_R-J_Hq*Su0_H)*Psi*model.keep+...
          Psi'*Su0_R'*Jc_R'*Jc_R*Su0_R*Psi*model.wc_R+Psi'*Su0_H'*Jc_H'*Jc_H*Su0_H*Psi*model.wc_H+...
          Psi'*Su0_R'*Tmax'*Tmax*Su0_R*Psi*model.wtheta_R+Psi'*Su0_R'*Tmin'*Tmin*Su0_R*Psi*model.wtheta_R+Psi'*R*Psi) \ (-Psi'*Suq'*(p(:)-pd(:))*model.wi_R-Psi'*Su0_R' *J_Rq'* f(:) * model.track - ...
          Psi'*(J_Rq*Su0_R-J_Hq*Su0_H)'*(X_R(:)-X_H(:))*model.keep - Psi'*Su0_R'*Jc_R'*fc_R*model.wc_R - Psi'*Su0_H'*Jc_H'*fc_H*model.wc_H - ...
          Psi'*Su0_R'*Tmax'*fmax_R*model.wtheta_R-Psi'*Su0_R'*Tmin'*fmin_R*model.wtheta_R - Psi'*R*(u_R-u_zero(:)));
    du = Psi*dw;
    %Estimate step size with backtracking line search method
	alpha = 1;
    cost0 = norm(f(:))^2 * model.track + (u_R-u_zero(:))'*R*(u_R-u_zero(:)) + (X_R(:)-X_H(:))'*(X_R(:)-X_H(:))*model.keep + ...
            norm(fc_R).^2*model.wc_R+norm(fc_H).^2*model.wc_H + model.wtheta_R*(norm(fmax_R)^2+norm(fmin_R)^2)+  norm(p(:)-pd())^2*model.wi_R;
    while 1
        try
		utmp = u_R + du * alpha;
		[xtmp_R, xtmp_H, xctmp_R,xctmp_H, ~, ~, ~, ~, ~, ~,~, ~] = fwdDyn(x0_R, x0_H, utmp, u_H, model, model_R, model_H);
        Qtmp_R = xtmp_R(:);
        ptmp= Qtmp_R(idq,1);
        Xtmp_R = fkin(xtmp_R, model_R, model.nbData);
        Xtmp_H = fkin(xtmp_H, model_H, model.nbData);
		ftmp = Xtmp_R(:) - xd;	
	    [fc0tmp_R, ~] = CoMcheck(xctmp_R, Jc00_R, model_R);
        [fc0tmp_H, ~] = CoMcheck(xctmp_H, Jc00_H, model_H);
        fctmp_R = fc0tmp_R(idc_R);
        fctmp_H = fc0tmp_H(idc_H);
        fmintmp_R = -(abs(T_thets*xtmp_R(:)-X_min_A) - (T_thets*xtmp_R(:)-X_min_A))/2;
        fmaxtmp_R = (abs(T_thets*xtmp_R(:) - X_max_A)+(T_thets*xtmp_R(:) - X_max_A))/2;
        cost = norm(ftmp(:))^2 * model.track+  (utmp-u_zero(:))'*R*(utmp-u_zero(:)) + (Xtmp_R(:)-Xtmp_H(:))'*(Xtmp_R(:)-Xtmp_H(:))*model.keep + ...
            norm(fctmp_R).^2*model.wc_R+norm(fctmp_H).^2*model.wc_H + model.wtheta_R*(norm(fmaxtmp_R)^2+norm(fmintmp_R)^2)+norm(ptmp(:)-pd())^2*model.wi_R;
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
        catch ME
            alpha = alpha * 0.5;
            if alpha < 1e-25
               stuck=1; 
               break;
            end
            continue;
        end
	end
	u_R = u_R + du * alpha;
    w= w + dw * alpha;
	if norm(du)*alpha/norm(u_R) < 1E-4 || stuck==1
        break; %Stop iLQR when solution is reached
	end
end
if stuck==1
    n=2e7;
    u_R=zeros(size(u_R));
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);
[x_R, x_H, xc_R,xc_H] = fwdDyn(x0_R, x0_H, u_R, u_H, model, model_R, model_H);
%Log data
u_R = reshape(u_R,model_R.nbVarX+2,[]); 
f = u_R(end-1:end,:);
u_R(end-1:end,:)=[];
%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colRob = lines(5);
tl = 1:model.nbData;
h=figure('position',[10,10,1200,1200],'color',[1,1,1]); hold on; axis off;
v = VideoWriter('./SWITCH.avi');
v.FrameRate = 10;
v.Quality = 100;
xlim([-1 3])
alarm =1;
open(v)
for t=1:1:length(tl)
        clf(h)
        axis off;
        xlim([-1 2])
        ylim([-.5 2.5])
        f_H = fkine00(x_H(1:model_H.nbVarX,t), model_H.L) + model_H.base; %End-effector position
		cTmp_R = colRob(2,:);
		plotArm2(x_H(1:model_H.nbVarX,tl(t)), model_R.L, [model_H.base; -10], .07, .3, 0.1,1, cTmp_R);
        hold on;
		plotCoM(xc_H(:,tl(t)), .03, max(cTmp_R-.1,0));
        hold on;
        plotRobotHead(x_H(1:model_H.nbVarX,tl(t)), model_H.L, [model_H.base; -10], model_H.H*0.121, cTmp_R);
        hold on;
        cTmp_H = colRob(1,:);
		plotArm2(x_R(1:model_R.nbVarX,tl(t)), model_R.L, [model_R.base; -10], .05, .2,.07,-1, cTmp_H);
        hold on;
		plotCoM(xc_R(:,tl(t)), .025, max(cTmp_H-.1,0));
        hold on;
        plotHumanHead(x_R(1:model_R.nbVarX,tl(t)), model_R.L, [model_R.base; -10], model_R.H*0.121, cTmp_H);
        hold on;
        plot([model_H.base(1)-.5 model_R.base(1)+.5],[-.08 -0.08], 'k-','linewidth',2)
        hold on;
        if mod(t,5)==0
            alarm = alarm*-1;
        end
        if alarm ==1
            for joint = 1:3
                if model_H.r_vec(1,joint)*1e5 > 100
                  scatter(f_H(1,joint),f_H(2,joint),200,'MarkerEdgeColor',colRob(3,:),'MarkerFaceColor',colRob(3,:),'MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
                end
            end
        end
        hold on
        drawnow;
		frame = getframe(gcf);
		writeVideo(v,frame);
        pause(0.05)
end
axis equal; 
close(v);


end

%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward dynamics
function [x_R, x_H, xc_R,xc_H, Su_R, Su_H, J_Rq, J_Hq, J_Rdq, J_Hdq,Jc_R, Jc_H] = fwdDyn(x_R, x_H, u_R, u_H, model, model_R, model_H)
    L_R=model_R.L;
    m_R=model_R.m;
    kv_R=model_R.kv;
    c_R=model_R.C;
    L_H=model_H.L;
    m_H=model_H.m;
    kv_H=model_H.kv;
    c_H=model_H.C;
    dt=model.dt;
    g=model.g;
    
    
	nbDOFs_R = length(L_R);
    nbDOFs_H = length(L_H);
    nbInp = nbDOFs_R + 2;
	nbData = (size(u_R,1) / nbInp) + 1;
	Tm_R1 = triu(ones(nbDOFs_R)) .* repmat(m_R, nbDOFs_R, 1);
    tm_R = Tm_R1 - diag(m_R) + diag(m_R.*c_R.^2);
	Tm_R= Tm_R1 - diag(m_R) + diag(m_R.*c_R);
    
    
    Tm_H1 = triu(ones(nbDOFs_H)) .* repmat(m_H, nbDOFs_H, 1);
    tm_H = Tm_H1 - diag(m_H) + diag(m_H.*c_H.^2);
	Tm_H= Tm_H1 - diag(m_H) + diag(m_H.*c_H);
	
    Su_R = zeros(2*(nbDOFs_R)*nbData, nbInp*(nbData-1));
    Su_H = zeros(2*(nbDOFs_H)*nbData, nbInp*(nbData-1));
	
	%Precomputation of mask (in tensor form)
	S1_R = zeros(nbDOFs_R,nbDOFs_R,nbDOFs_R);
	for k=1:nbDOFs_R
		for i=1:nbDOFs_R
			for j=1:nbDOFs_R
				S1_R(k,i,j) = [logical(i==j)-logical(k==j)];
			end
		end
    end
    
    S1_H = zeros(nbDOFs_H,nbDOFs_H,nbDOFs_H);
	for k=1:nbDOFs_H
		for i=1:nbDOFs_H
			for j=1:nbDOFs_H
				S1_H(k,i,j) = [logical(i==j)-logical(k==j)];
			end
		end
    end

    
    J_Rq=[];
    J_Hq=[];
    J_Rdq=[];
    J_Hdq=[];
    Jc_R = [];
    Jc_H = [];
    
    
     X_R = fkineCoM00(x_R(1:model_R.nbVarX,1),model_R);
     xc_R(:,1) = model_R.m*X_R(:,2:end)'/sum(model_R.m);

     X_H = fkineCoM00(x_H(1:model_H.nbVarX,1),model_H);
     xc_H(:,1) = model_H.m*X_H(:,2:end)'/sum(model_H.m);
	for t=1:nbData-1

		J_R = [cos(x_R(1:nbDOFs_R,t))'; sin(x_R(1:nbDOFs_R,t))'] * diag(L_R);
		Jctmp_R = J_R*diag(sum(Tm_R,2))/sum(model_R.m);
        G_R = -sum(Tm_R,2) .* L_R' .* sin(x_R(1:nbDOFs_R,t)) * g;
		M_R = (L_R' * L_R) .* cos(x_R(1:nbDOFs_R,t) - x_R(1:nbDOFs_R,t)') .* (tm_R.^.5 * tm_R.^.5') + diag(model_R.I);
		C_R = -(L_R' * L_R) .* sin(x_R(1:nbDOFs_R,t) - x_R(1:nbDOFs_R,t)') .* (tm_R.^.5 * tm_R.^.5');

        
        J_H = [cos(x_H(1:nbDOFs_H,t))'; sin(x_H(1:nbDOFs_H,t))'] * diag(L_H);
		Jctmp_H = J_H*diag(sum(Tm_H,2))/sum(model_H.m);
        G_H = -sum(Tm_H,2) .* L_H' .* sin(x_H(1:nbDOFs_H,t)) * g;
		M_H =  (L_H' * L_H) .* cos(x_H(1:nbDOFs_H,t) - x_H(1:nbDOFs_H,t)') .* (tm_H.^.5 * tm_H.^.5') + diag(model_H.I);
		C_H = -(L_H' * L_H) .* sin(x_H(1:nbDOFs_H,t) - x_H(1:nbDOFs_H,t)') .* (tm_H.^.5 * tm_H.^.5');

        
		%Computation in tensor form of derivatives dJ,dG,dM,dC 
		dJ_tmp_R = [-sin(x_R(1:nbDOFs_R,t))'; cos(x_R(1:nbDOFs_R,t))'] * diag(L_R);
		dJ_tmp_R = kron(dJ_tmp_R, [1,zeros(1,nbDOFs_R)]);
		dJ_R = reshape(dJ_tmp_R(:,1:nbDOFs_R^2), [2,nbDOFs_R,nbDOFs_R]);
		dG_R = diag(-sum(Tm_R,2) .* L_R' .* cos(x_R(1:nbDOFs_R,t)) * g);
		dM_tmp_R = (L_R' * L_R) .* sin(x_R(1:nbDOFs_R,t) - x_R(1:nbDOFs_R,t)') .* (tm_R.^.5 * tm_R.^.5');
		dC_tmp_R = (L_R' * L_R) .* cos(x_R(1:nbDOFs_R,t) - x_R(1:nbDOFs_R,t)') .* (tm_R.^.5 * tm_R.^.5');
		dM_R = repmat(dM_tmp_R, [1,1,nbDOFs_R]) .* S1_R;
		dC_R = repmat(dC_tmp_R, [1,1,nbDOFs_R]) .* S1_R;
        
        
        dJ_tmp_H = [-sin(x_H(1:nbDOFs_H,t))'; cos(x_H(1:nbDOFs_H,t))'] * diag(L_H);
		dJ_tmp_H = kron(dJ_tmp_H, [1,zeros(1,nbDOFs_H)]);
		dJ_H = reshape(dJ_tmp_H(:,1:nbDOFs_H^2), [2,nbDOFs_H,nbDOFs_H]);
		dG_H = diag(-sum(Tm_H,2) .* L_H' .* cos(x_H(1:nbDOFs_H,t)) * g);
		dM_tmp_H = (L_H' * L_H) .* sin(x_H(1:nbDOFs_H,t) - x_H(1:nbDOFs_H,t)') .* (tm_H.^.5 * tm_H.^.5');
		dC_tmp_H = (L_H' * L_H) .* cos(x_H(1:nbDOFs_H,t) - x_H(1:nbDOFs_H,t)') .* (tm_H.^.5 * tm_H.^.5');
		dM_H = repmat(dM_tmp_H, [1,1,nbDOFs_H]) .* S1_H;
		dC_H = repmat(dC_tmp_H, [1,1,nbDOFs_H]) .* S1_H;
        

        tau_H = u_H(:,t);
        tau_R = u_R((t-1)*(model_R.nbVarU+2)+1:(t-1)*(model_R.nbVarU+2)+nbDOFs_R);
        Fext = u_R((t-1)*(model_R.nbVarU+2)+nbDOFs_R+1:(t-1)*(model_R.nbVarU+2)+nbDOFs_R + 2); %External force at end-effector

		ddq_B = pinv(M_H)* (tau_H + J_H'* Fext + G_H + C_H * x_H(nbDOFs_H+1:2*nbDOFs_H,t).^2- x_H(nbDOFs_H+1:2*nbDOFs_H,t) * kv_H) ; %With external force and joint damping
        ddq_A = pinv(M_R)* (tau_R - J_R'* Fext + G_R + C_R * x_R(nbDOFs_R+1:2*nbDOFs_R,t).^2- x_R(nbDOFs_R+1:2*nbDOFs_R,t) * kv_R) ; %With external force and joint damping

        x_H(:,t+1) = x_H(:,t) + [x_H(nbDOFs_H+1:2*nbDOFs_H,t); ddq_B] * dt;
        x_R(:,t+1) = x_R(:,t) + [x_R(nbDOFs_R+1:2*nbDOFs_R,t); ddq_A] * dt;
		
        X_R = fkineCoM00(x_R(1:model_R.nbVarX,t+1),model_R);
        xc_R(:,t+1) = model_R.m*X_R(:,2:end)'/sum(model_R.m);

        X_H = fkineCoM00(x_H(1:model_H.nbVarX,t+1),model_H);
        xc_H(:,t+1) = model_H.m*X_H(:,2:end)'/sum(model_H.m);

		%Compute local linear systems        
		invM_R = pinv(M_R);
		for j=1:nbDOFs_R
			A21_R(:,j) = -invM_R * dM_R(:,:,j) * invM_R * (tau_R - J_R'* Fext) - invM_R * dJ_R(:,:,j)' * Fext - invM_R * dM_R(:,:,j) * invM_R * G_R + ...
				invM_R * dG_R(:,j) - invM_R * dM_R(:,:,j) * invM_R * (C_R * x_R(nbDOFs_R+1:2*nbDOFs_R,t).^2 - x_R(nbDOFs_R+1:2*nbDOFs_R,t) * kv_R)+ invM_R * dC_R(:,:,j) * x_R(nbDOFs_R+1:2*nbDOFs_R,t).^2;
        end
		
        
       
		invM_H = pinv(M_H);
		for j=1:nbDOFs_H
			A21_H(:,j) = -invM_H * dM_H(:,:,j) * invM_H * (tau_H + J_H'* Fext) + invM_H * dJ_H(:,:,j)' * Fext - invM_H * dM_H(:,:,j) * invM_H * G_H + ...
				invM_H * dG_H(:,j) - invM_H * dM_H(:,:,j) * invM_H * (C_H * x_H(nbDOFs_H+1:2*nbDOFs_H,t).^2 - x_H(nbDOFs_H+1:2*nbDOFs_H,t) * kv_H) + invM_H * dC_H(:,:,j) * x_H(nbDOFs_H+1:2*nbDOFs_H,t).^2;
		end
		%Linear systems with all components
        
        
        A_H(:,:,t) = [eye(nbDOFs_H), eye(nbDOFs_H)*dt;A21_H * dt, eye(nbDOFs_H) + [invM_H * (2*C_H * diag([x_H(nbDOFs_H+1:2*nbDOFs_H,t)])-eye(nbDOFs_H)*kv_H)]*dt];
        A_R(:,:,t) = [eye(nbDOFs_R), eye(nbDOFs_R)*dt;A21_R * dt, eye(nbDOFs_R) + [invM_R * (2*C_R * diag([x_R(nbDOFs_R+1:2*nbDOFs_R,t)])-eye(nbDOFs_R)*kv_R)]*dt];
        
        B_H(:,:,t) = [zeros(nbDOFs_H,nbDOFs_R+2);zeros(nbDOFs_H,nbDOFs_R) invM_H*J_H'*dt];
        B_R(:,:,t) = [zeros(nbDOFs_R,nbDOFs_R+2);invM_R*dt -invM_R*J_R'*dt];
        
        
        Su_R(2*(nbDOFs_R)*t+1:2*(nbDOFs_R)*(t+1),:) = A_R(:,:,t) * Su_R(2*(nbDOFs_R)*(t-1)+1:2*(nbDOFs_R)*(t),:);
        Su_R(2*(nbDOFs_R)*t+1:2*(nbDOFs_R)*(t+1), nbInp*(t-1)+1:nbInp*t) = B_R(:,:,t);
        
        Su_H(2*(nbDOFs_H)*t+1:2*(nbDOFs_H)*(t+1),:) = A_H(:,:,t) * Su_H(2*(nbDOFs_H)*(t-1)+1:2*(nbDOFs_H)*(t),:);
        Su_H(2*(nbDOFs_H)*t+1:2*(nbDOFs_H)*(t+1), nbInp*(t-1)+1:nbInp*t) = B_H(:,:,t);
        
        J_Rq=blkdiag(J_Rq,[J_R zeros(2,nbDOFs_R)]);
         J_Hq=blkdiag(J_Hq,[J_H zeros(2,nbDOFs_H)]);

        
         J_Rdq=blkdiag(J_Rdq,[zeros(2,nbDOFs_R) J_R]);
         J_Hdq=blkdiag(J_Hdq,[zeros(2,nbDOFs_H) J_H]);
         
         
         Jc_R =blkdiag(Jc_R,[Jctmp_R(1,:) zeros(1,model_R.nbVarX)]);
         Jc_H =blkdiag(Jc_H,[Jctmp_H(1,:) zeros(1,model_H.nbVarX)]);
            
    end 

    J_R = [cos(x_R(1:nbDOFs_R,end))'; sin(x_R(1:nbDOFs_R,end))'] * diag(L_R);
    Jctmp_R = J_R*diag(sum(Tm_R,2))/sum(model_R.m);
    J_H = [cos(x_H(1:nbDOFs_H,end))'; sin(x_H(1:nbDOFs_H,end))'] * diag(L_H);
    Jctmp_H = J_H*diag(sum(Tm_H,2))/sum(model_H.m);
    J_Rdq=blkdiag(J_Rdq,[zeros(2,nbDOFs_R) J_R]);
    J_Hdq=blkdiag(J_Hdq,[zeros(2,nbDOFs_H) J_H]);
    J_Rq=blkdiag(J_Rq,[J_R zeros(2,nbDOFs_R)]);
    J_Hq=blkdiag(J_Hq,[J_H zeros(2,nbDOFs_H)]);
    Jc_R =blkdiag(Jc_R,[Jctmp_R(1,:) zeros(1,model_R.nbVarX)]);
    Jc_H =blkdiag(Jc_H,[Jctmp_H(1,:) zeros(1,model_H.nbVarX)]);
end

%Forward kinematics for all robot articulations (in the agent coordinate system)
function f = fkine00(x, L)
	T2 = tril(ones(size(x,1))) .* repmat(L, size(x,1), 1);
	f = [T2 * sin(x), ...
	    -T2 * cos(x)]'; 
	f = [zeros(2,1), f];
end

%Forward kinematics for all robot articulations (in the human coordinate
%system) for all times
function f = fkin(x, model,nbData)
f=[];
	for t=1:nbData
        f0=fkine00(x(1:model.nbVarX,t),model.L);
        ft=f0(:,end) + model.base;
        f=[f;ft];
    end
end

%%%%%%%%%%%%%%%%%%%%%%
% Position of each link CoM in the task space

function f = fkineCoM00(x, model)
	T2 = tril(ones(size(x,1))) .* repmat(model.L, size(x,1), 1);
    T2 = T2 - diag(model.L) + diag(model.L.*model.C);
	f = [T2 * sin(x), ...
	    -T2 * cos(x)]'; 
	f = [zeros(2,1), f]+model.base;
end

%%%%%%%%%%%%%%%%%%%%%%
% Modify the parameters of CoM cost according to its position w.r.t. the
% feasibile region

function [fc, Jc] = CoMcheck(xc, Jc0, model)
Jc=Jc0;
 for t = 1:size(xc,2)
     in_boundry = logical(abs(xc(1,t)-model.MuCoM) < model.szCoM);
     if in_boundry
        fc(t,1) = 0; 
        Jc(t,:) = zeros(1,size(Jc0(t,:),2));
     else
        fc(t,1) = xc(1,t) - (model.MuCoM + sign(xc(1,t)-model.MuCoM)*model.szCoM);
     end
     
 end
end
% Draw the human's head
function h =  plotHumanHead(q, L, p, sz,facecolor)

edgecolor = [1,1,1];
alpha = 1;


nbSegm = 30;
t = linspace(0,2*pi,nbSegm);
Head(1,:) = [sz/2.*sin(t)];
Head(2,:) = [sz/2.*cos(t)];
Head(3,:) = zeros(1,nbSegm);

R1 = [cos(q(1)-pi/2) -sin(q(1)-pi/2) 0; sin(q(1)-pi/2) cos(q(1)-pi/2) 0; 0 0 0];
p1 = R1*[L(1);0;0] + p;
R2 = [cos(q(2)-pi/2) -sin(q(2)-pi/2) 0; sin(q(2)-pi/2) cos(q(2)-pi/2) 0; 0 0 0];
p2 = R2*[L(2);0;0] + p1;
R3 = [cos(q(3)-pi/2) -sin(q(3)-pi/2) 0; sin(q(3)-pi/2) cos(q(3)-pi/2) 0; 0 0 0];
p3 = R3*[L(3)+sz/2 + sz/3;0;0] + p2;
p_head = R3*Head + p3;
patch(p_head(1,:),p_head(2,:),repmat(p(3),1,nbSegm),facecolor,'edgeColor',edgecolor,'linewidth',3,'edgealpha',alpha,'facealpha',alpha); 

end

% Draw the human's head
function h =  plotRobotHead(q, L, p, sz,facecolor)

edgecolor = [1,1,1];
alpha = 1;

Square = [-.5 .5 .5 -.5; 1.618/2 1.618/2 -1.618/2 -1.618/2;0 0 0 0]*sz;
% h = plotArmBasis(p, sz, facecolor, edgecolor, alpha);
R1 = [cos(q(1)-pi/2) -sin(q(1)-pi/2) 0; sin(q(1)-pi/2) cos(q(1)-pi/2) 0; 0 0 0];
p1 = R1*[L(1);0;0] + p;
R2 = [cos(q(2)-pi/2) -sin(q(2)-pi/2) 0; sin(q(2)-pi/2) cos(q(2)-pi/2) 0; 0 0 0];
p2 = R2*[L(2);0;0] + p1;
R3 = [cos(q(3)-pi/2) -sin(q(3)-pi/2) 0; sin(q(3)-pi/2) cos(q(3)-pi/2) 0; 0 0 0];
p3 = R3*[L(3)+sz/2 + sz/3;0;0] + p2;
p_head = R3*Square + p3;
patch(p_head(1,:),p_head(2,:),repmat(p(3),1,4),facecolor,'edgeColor',edgecolor,'linewidth',3,'edgealpha',alpha,'facealpha',alpha); 

end

