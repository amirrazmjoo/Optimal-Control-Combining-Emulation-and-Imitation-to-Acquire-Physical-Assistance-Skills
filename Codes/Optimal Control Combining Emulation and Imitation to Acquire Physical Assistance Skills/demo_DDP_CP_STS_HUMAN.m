function [u_H,w,x_H, F,n] = demo_DDP_CP_STS_HUMAN(model,model_H,x0_H)
% The human assistance prediction, solved by iLQR and modeled by Lagrangian formulation
% Sylvain Calinon and Amirreza Razmjoo, 2021




%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Optimization model
if nargin<1
    model.dt = .01; %Step size
    model.nbData = 50; %Number of datapoints
    model.nbIter = 200; %Maximum number of iterations for iLQR
    model.Mu = [pi*ones(3,1);zeros(3,1)]; %Target positions of hip, knee and ankle joints and their corresponding velocities
    model.wq = 1E3; %Weight of being at Stand position (model.Mu)
    model.wi = 10;% Weight of imitiation cost
    model.wc = 1E4; %Weight of the human's CoM position to be at stable position (foot support) at the end of the task
    model.wtheta=1E2; %Weight of joint limits cost
    model.g= 9.81; %Gravity
    model.nbDemos = 15; %Number of demonstrations used for warm starting and extracting imitation fearures
end
% Human model
if nargin<2 
    model_H.nbVarX = 5;%Number of links in the human model
    model_H.nbDofs = model_H.nbVarX; % Human's number of degrees of freedom
    model_H.nbVarU=model_H.nbVarX; % Number of the human's joints
    model_H.kv=10; % Damping ratios at the human's joints
    model_H.xmin= [3*pi/4;0;-pi/2;-pi;0]; %Minimum joints limits (first one defined w.r.t. downside of the vertical line, and the other are relative angeles)
    model_H.xmax= [5*pi/4;pi/2;0;-pi/4;3*pi/4];%Maximum joints limits (first one defined w.r.t. downside of the vertical line, and the other are relative angeles)
    model_H.H = 1.8;% Height of the human (m)
    model_H.M = 80; % Mass of the human (kg)
    model_H.L = model_H.H(1,:)*[250 240 300 183 160]/1000; % Length of each human link
    model_H.foot= model_H.H(1,:)*50/1000; %Length of the human's foot
    model_H.C = [0.532 0.5 0.53 0.43 0.41]; %Each human link CoM (normalized by the length of each link)
    model_H.m = model_H.M*[46.5*2 100*2 508 27*2 16*2]/1000; %Mass of each human link
    model_H.I = [0.0405*2 0.1502*2 1.3080 0.02*2 0.076*2]; %Links moment of inertias
    model_H.MuCoM = model_H.foot/2; %Target point for center of mass
    model_H.szCoM = model_H.foot/2*0.8; %CoM allowed width
    model_H.nbFct = 5; %Number of basis functuin used for control and movement primitivs
    model_H.r_vec = [.1000 .1000 .1000 1 1 1 1]*1e-5; % [ankle, knee, hip, shoulder, elbow, Horizontal external force, Vertical external force]. Only the first three should be changed in the range of .1 to 1000.
end
if nargin<3
    x0_H = [pi;3*pi/2;pi;pi/4;pi/2;zeros(model_H.nbVarX,1)];
end
R0 =[];
for i=1:length(model_H.r_vec)
    R0 = blkdiag(R0,model_H.r_vec(i));
end
R = sparse(kron(eye(model.nbData-1),R0)); %Control weight matrix (at trajectory level)



% Concatenating all joint limits
T_thets = kron(eye(model.nbData),[inv(tril(ones(model_H.nbVarX,model_H.nbVarX))) zeros(model_H.nbVarX,model_H.nbVarX)]); % Transfer matrix to convert absolute angles (used in dynamical motion) to relative angles (joint limits defined in this format)
X_min = kron(ones(model.nbData,1),model_H.xmin);
X_max = kron(ones(model.nbData,1),model_H.xmax);



% Defining important Ids to consider in calculating the gradinet. To speed
% up the code
idq0 = [repmat(1:model_H.nbVarX,model.nbData,1) + ([0:model.nbData-1]*2*model_H.nbVarX)']';
idq =  idq0(:); %features to follow (only joint angles)

idx0 = [1:3 model_H.nbVarX+[1:3]];
idc = model.nbData-2:model.nbData; % Times we want the CoM of the human be in foot support
idx = (model.nbData - 1) * model_H.nbVarX*2 + idx0; %Features we consider at the end of the task (joint angles of the human's ankle, knee and hip and their corresponding values)








% Transfer matrix of control primitives
nbFct = model_H.nbFct;
t = linspace(0, 1, model.nbData-1);
tMu = linspace(t(1), t(end), nbFct);
phi = zeros(model.nbData-1, nbFct);
for i=1:nbFct
    phi(:,i) = gaussPDF(t, tMu(i), 2E-2);
    % 	phi(:,i) = mvnpdf(t', tMu(i), 2E-2);
end
phi = phi ./ repmat(sum(phi,2),1,nbFct); %Optional rescaling
Psi = kron(phi, eye((model_H.nbVarU+2)));


% Transfer matrix of movement primitives
tx = linspace(0, 1, model.nbData);
tMux = linspace(tx(1), tx(end), nbFct);
phix = zeros(model.nbData, nbFct);
for i=1:nbFct
    phix(:,i) = gaussPDF(tx, tMux(i), 2E-2);
    % 	phi(:,i) = mvnpdf(t', tMu(i), 2E-2);
end
phix = phix ./ repmat(sum(phix,2),1,nbFct); %Optional rescaling
Psix = kron(phix, eye((model_H.nbVarX)));


%% warm starting and imitation trajectory (using k-NN with k=1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Read demonstrations
demos = load('./STS_demonstrations.mat').sts_demo;
demo_input=[];
demo_wuH = [];
demo_wxH = [];
for k = 1 : model.nbDemos
    demo_input = [demo_input;demos{k}.model_H.r_vec(1:3)*1e5 demos{k}.model_H.H demos{k}.model_H.M];
    demo_wuH = [demo_wuH;demos{k}.wu_H'];
    demo_wxH = [demo_wxH;demos{k}.wx_H'];
end

%Normalize the data
demo_input(:,1:3) = log10(demo_input(:,1:3));
Mean_demo = mean(demo_input);
std_demo = std(demo_input);
current_input = [log10(model_H.r_vec(1:3)*1e5) model_H.H model_H.M];
concat_data = [current_input;demo_input];
norm_data = (concat_data-Mean_demo)./std_demo;
%Find the nearest neighbor
distances = dist(norm_data);
[~, NN_idx] = min(distances(2:end,1));
w0x = demo_wxH(NN_idx,:)';
w0u = demo_wuH(NN_idx,:)';



% Keeping only the movement primitives of the joint angles (w0x includes movement primitives of the joint angles and velocities)
wx = reshape(w0x,2*model_H.nbVarX,[]);
wx(model_H.nbVarX+1:end,:)=[];

% Extracting the desired imitation trajectory
pd = Psix*wx(:);

%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = x0_H;
w=w0u;
u = Psi*w; %Torque commands

for n=1:model.nbIter
    %System evolution
    [x, xc, A, B, Su0, Jc00] = fwdDyn(x0, u, model, model_H);
    Q = x(:);
    p= Q(idq,1); %Imitation feature at the current iteration
    Su = Su0(idx,:);
    Suq =Su0 (idq,:);
    f = x(idx0,end) - model.Mu;
    [fc0, Jc0] = CoMcheck(xc, Jc00, model_H); %Check if we need to consider stability cost (if the system already be in the feasible region, we do not need to consider this)
    fc = fc0(idc);
    Jc = Jc0 (idc,:);
    % Joint limits deviations (if the system was in the feasible region, the value of proceeding formulations would be zero)
    fmin = -(abs(T_thets*x(:)-X_min) - (T_thets*x(:)-X_min))/2;
    fmax = (abs(T_thets*x(:) - X_max)+(T_thets*x(:) - X_max))/2;
    Tmax = diag(sign(fmax))*T_thets;
    Tmin = -diag(sign(fmin)) * T_thets;
    
    % Update value of the coefficients of the CPs
    dw = (Psi'*Suq' * Suq *Psi*model.wi+Psi'*Su' * Su *Psi*model.wq + Psi'*Su0'*Jc'*Jc*Su0*Psi*model.wc+ Psi'*Su0'*Tmax'*Tmax*Su0*Psi*model.wtheta+Psi'*Su0'*Tmin'*Tmin*Su0*Psi*model.wtheta+Psi'*R*Psi) \ (-Psi'*Suq' * (p-pd) * model.wi-Psi'*Su' * f(:) * model.wq - Psi'*Su0'*Jc'*fc*model.wc- Psi'*Su0'*Tmax'*fmax*model.wtheta-Psi'*Su0'*Tmin'*fmin*model.wtheta -Psi'*R*u);
    du = Psi *dw;
    %Estimate step size with backtracking line search method
    alpha = 1;
    cost0 = norm(f(:))^2 * model.wq + u'*R*u + norm(fc(:))^2*model.wc+model.wtheta*(norm(fmax)^2+norm(fmin)^2)+norm(p(:)-pd)^2*model.wi;
    while 1
        try
            utmp = u + du * alpha;
            [xtmp, xctmp] = fwdDyn(x0, utmp, model, model_H);
            Qtmp = xtmp(:);
            ptmp= Qtmp(idq,1);
            fmintmp = -(abs(T_thets*xtmp(:)-X_min) - (T_thets*xtmp(:)-X_min))/2;
            fmaxtmp = (abs(T_thets*xtmp(:) - X_max)+(T_thets*xtmp(:) - X_max))/2;
            ftmp = xtmp(idx0,end) - model.Mu;
            [fc0tmp, ~] = CoMcheck(xc, Jc00, model_H);
            fctmp = fc0tmp(idc);
            cost = norm(ftmp(:))^2 * model.wq + utmp' *R*utmp + norm(fctmp(:))^2*model.wc +model.wtheta*(norm(fmaxtmp)^2+norm(fmintmp)^2)+norm(ptmp(:)-pd)^2*model.wi;
            if cost < cost0 || alpha < 1E-4
                break;
            end
            alpha = alpha * 0.5;
        catch ME
            alpha = alpha * 0.5;
        end
    end
    %Update the current values
    u = u + du * alpha;
    w = w + dw * alpha;
    %Terminating criteria
    if norm(du * alpha)/norm(u) < 1E-4
        break; %Stop iLQR when solution is reached
    end
end
disp(['iLQR converged for only system B in ' num2str(n) ' iterations.']);
[x, xc] = fwdDyn(x0, u, model, model_H); %Update the states of the system after last step

%Log data
x_H = x;
xc_H = xc;
u = reshape(u, model_H.nbVarU+2, model.nbData-1);
u_H = u(1:model_H.nbVarU,:);
F =  u(model_H.nbVarU+1:end,:);

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Timeline plot
colRob = lines(1);
h=figure('position',[10,10,1200,1200],'color',[1,1,1]); hold on; axis off;
alarm = 1;

for t=1:model.nbData
    clf(h)
    axis off;
    xlim([-1 2])
    ylim([-.5 2.5])
    f = fkine00(x_H(1:model_H.nbVarX,t), model_H.L); %End-effector position
    fc = fkineCoM00(x_H(1:model_H.nbVarX,t), model_H);
    plotArm2(x_H(1:model_H.nbVarX,t), model_H.L, [0;0;-10], .05, .2,.07,1, colRob);
    hold on;
    plotCoM(xc_H(:,t), .025, max(colRob-.1,0));
    hold on;
    plotHumanHead(x_H(1:model_H.nbVarX,t), model_H.L, [0;0; -10], model_H.H * 0.121, colRob);
    hold on;
    plot([-.5 1.5],[-.08 -0.08], 'k-','linewidth',2)
    hold on
    if mod(t,5)==0
        alarm = alarm*-1;
    end
    if alarm ==1
        for joint = 1:3
            if model_H.r_vec(joint)*1e5 > 100
                scatter(f_B(1,joint),f_B(2,joint),200,'MarkerEdgeColor',[1,0,0],'MarkerFaceColor',[1,0,0],'MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
            end
        end
    end
    drawnow;
    pause(0.01);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward dynamics
function [x, xc, A, B, Su, Jc] = fwdDyn(x, u, model, model_H)
L=model_H.L;
m=model_H.m;
c = model_H.C;
dt=model.dt;
g=model.g;
kv=model_H.kv;
nbDOFs = length(L);
nbInp=nbDOFs+2;
nbData = (size(u,1) / nbInp) + 1;
Tm0 = triu(ones(nbDOFs)) .* repmat(m, nbDOFs, 1);
tm = Tm0 - diag(m)+diag(m.*c.^2);
Tm = Tm0 - diag(m)+diag(m.*c);
Su = zeros(2*nbDOFs*nbData, nbInp*(nbData-1));

%Precomputation of mask (in tensor form)
S1 = zeros(nbDOFs,nbDOFs,nbDOFs);
for k=1:nbDOFs
    for i=1:nbDOFs
        for j=1:nbDOFs
            S1(k,i,j) = [logical(i==j)-logical(k==j)];
        end
    end
end
X = fkineCoM00(x(1:nbDOFs,1),model_H);
xc(:,1) = m*X(:,2:end)'/sum(m);
Jc=[];
for t=1:nbData-1
    
    %Computation in matrix form of J,G,M,C
    J = [cos(x(1:nbDOFs,t))'; sin(x(1:nbDOFs,t))'] * diag(L);
    Jctmp = J*diag(sum(Tm,2))/sum(m);
    G = -sum(Tm,2).*L'.* sin(x(1:nbDOFs,t)) * g;
    M =  (L' * L) .* cos(x(1:nbDOFs,t) - x(1:nbDOFs,t)') .* (tm.^.5 * tm.^.5') +diag(model_H.I);
    C = -(L' * L) .* sin(x(1:nbDOFs,t) - x(1:nbDOFs,t)') .* (tm.^.5 * tm.^.5');
    
    
    
    
    %Computation in tensor form of derivatives dJ,dG,dM,dC
    dJ_tmp = [-sin(x(1:nbDOFs,t))'; cos(x(1:nbDOFs,t))'] * diag(L);
    dJ_tmp = kron(dJ_tmp, [1,zeros(1,nbDOFs)]);
    dJ = reshape(dJ_tmp(:,1:nbDOFs^2), [2,nbDOFs,nbDOFs]);
    dG = diag(-sum(Tm,2).*L' .* cos(x(1:nbDOFs,t)) * g);
    dM_tmp = (L' * L) .* sin(x(1:nbDOFs,t) - x(1:nbDOFs,t)') .* (tm.^.5 * tm.^.5');
    dC_tmp = (L' * L) .* cos(x(1:nbDOFs,t) - x(1:nbDOFs,t)') .* (tm.^.5 * tm.^.5');
    dM = repmat(dM_tmp, [1,1,nbDOFs]) .* S1;
    dC = repmat(dC_tmp, [1,1,nbDOFs]) .* S1;
    
    %Update pose
    u_t=u((t-1)*nbInp+1:t*nbInp);
    tau = u_t(1:nbDOFs);
    Fext = u_t(nbDOFs+1:nbDOFs+2); %External force at end-effector
    
    ddq = pinv(M)* (tau + J'* Fext + G + C * x(nbDOFs+1:2*nbDOFs,t).^2- x(nbDOFs+1:2*nbDOFs,t) * kv); %With external force and joint damping
    
    
    x(:,t+1) = x(:,t) + [x(nbDOFs+1:2*nbDOFs,t); ddq] * dt;
    X = fkineCoM00(x(1:nbDOFs,t+1),model_H);
    xc(:,t+1) = m*X(:,2:end)'/sum(m);
    
    %Compute local linear systems
    invM = pinv(M);
    for j=1:nbDOFs
        A21(:,j) = -invM * dM(:,:,j) * invM * (tau + J'* Fext) + invM * dJ(:,:,j)' * Fext - invM * dM(:,:,j) * invM * G + ...
            invM * dG(:,j) - invM * dM(:,:,j) * invM * (C * x(nbDOFs+1:2*nbDOFs,t).^2- x(nbDOFs+1:2*nbDOFs,t) * kv) + invM * dC(:,:,j) * x(nbDOFs+1:2*nbDOFs,t).^2;
    end
    
    %Linear systems with all components
    A(:,:,t) = [eye(nbDOFs), eye(nbDOFs)*dt; A21 * dt, eye(nbDOFs) + [invM * (2*C * diag([x(nbDOFs+1:2*nbDOFs,t)])-eye(nbDOFs)*kv)]*dt];
    B(:,:,t) = [zeros(nbDOFs,nbInp); invM*dt invM*J'*dt];
    Jc =blkdiag(Jc,[Jctmp(1,:) zeros(1,nbDOFs)]);
    
    Su(2*nbDOFs*t+1:2*nbDOFs*(t+1),:) = A(:,:,t) * Su(2*nbDOFs*(t-1)+1:2*nbDOFs*(t),:);
    Su(2*nbDOFs*t+1:2*nbDOFs*(t+1), nbInp*(t-1)+1:nbInp*t) = B(:,:,t);
    
    
    
    
    
end %t
J = [cos(x(1:nbDOFs,end))'; sin(x(1:nbDOFs,end))'] * diag(L);
Jctmp = J*diag(sum(Tm,2))/sum(m);
Jc =blkdiag(Jc,[Jctmp(1,:) zeros(1,nbDOFs)]);
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all robot articulations (in robot coordinate system)
function f = fkine00(x, L)
T2 = tril(ones(size(x,1))) .* repmat(L, size(x,1), 1);
f = [T2 * sin(x), ...
    -T2 * cos(x)]';
f = [zeros(2,1), f];
end

%%%%%%%%%%%%%%%%%%%%%%
% Position of each link CoM in the task space
function f = fkineCoM00(x, model_H)
T2 = tril(ones(size(x,1))) .* repmat(model_H.L, size(x,1), 1);
T2 = T2 - diag(model_H.L) + diag(model_H.L.*model_H.C);
f = [T2 * sin(x), ...
    -T2 * cos(x)]';
f = [zeros(2,1), f];
end


%%%%%%%%%%%%%%%%%%%%%%
% Modify the parameters of CoM cost according to its position w.r.t. the
% feasibile region
function [fc, Jc] = CoMcheck(xc, Jc0, model_H)
Jc=Jc0;
for t = 1:size(xc,2)
    in_boundry = logical(abs(xc(1,t)-model_H.MuCoM) < model_H.szCoM);
    if in_boundry
        fc(t,1) = 0;
        Jc(t,:) = zeros(1,size(Jc0(t,:),2));
    else
        fc(t,1) = xc(1,t) - (model_H.MuCoM + sign(xc(1,t)-model_H.MuCoM)*model_H.szCoM);
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