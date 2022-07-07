function [] = demo_DDP_CP_STS


%Optimization model
model.dt = .01; %Step size
model.nbData = 50; %Number of datapoints
model.nbIter = 200; %Maximum number of iterations for iLQR
model.g= 9.81; %Gravity
model.nbDemos = 15; %Number of demonstrations used for warm starting and extracting imitation fearures
%model parameters for the human assistance prediction
model.Mu = [pi*ones(3,1);zeros(3,1)]; %Target positions of hip, knee and ankle joints and their corresponding velocities
model.wq = 1E3; %Weight of being at Stand position (model.Mu)
model.wi = 1E1;% Weight of imitiation cost
model.wc = 1E4; %Weight of the human's CoM position to be at stable position (foot support) at the end of the task
model.wtheta=1E2; %Weight of joint limits cost
%Model paramerers for the robot controller
model.wi_R =1e2; %Cost of the imitation term
model.wr = 1E-5;
model.keep = 1E6;
model.track = 1E3;
model.wc_R = 1E5;
model.wc_H = 1E2;
model.wtheta_R = 1E4;




%The human model
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
model_H.base = [0;0];
model_H.nbFct = 5; %Number of basis functuin used for control and movement primitivs
x0_H = [pi;3*pi/2;pi;pi/4;pi/2;zeros(model_H.nbVarX,1)];
model_H.r_vec = [1.000 1.000 1.000 1 1 1 1]*1e-5; % [ankle, knee, hip, shoulder, elbow, Horizontal external force, Vertical external force]. Only the first three should be changed in the range of .1 to 1000.
EE_H=(model_H.L(1,:)*[sin(x0_H(1:model_H.nbVarX,1)) -cos(x0_H(1:model_H.nbVarX,1))])'; %position of the human's hand in the task space
%The robot model
model_R.nbVarX = 5; %State space dimension (q1,q2,q3, ...)
model_R.nbDofs = model_R.nbVarX;
model_R.nbVarU=model_R.nbVarX;%Inputes (tau1,tau2,...,tauN,Fx,Fy);
model_R.kv=10;
model_R.H = 1.8;
model_R.M = 80;
model_R.L = model_R.H*[250 240 300 183 160]/1000; %Link length
model_R.m = model_R.M*[46.5*2 100*2 508 27*2 16*2]/1000; %Link mass
model_R.foot= model_R.H*50/1000;
model_R.I = [0.0405*2 0.1502*2 1.3080 0.02*2 0.076*2];
model_R.xmin=[3/4;-.5;0;0;-2/4]*pi;
model_R.xmax=[5/4;0;1/2;1/2;0]*pi;
model_R.C(1,:) = [0.532 0.5 0.53 0.43 0.41];
model_R.base = [.7;0];
model_R.MuCoM = model_R.base(1,1) - model_R.foot/2; %Target point for center of mass
model_R.szCoM = model_R.foot/2*0.6; %CoM allowed width
model_R.nbFct = 8;
EE_R = model_H.base-model_R.base+EE_H; % Where the end_effector of the robot should be w.r.t. its base
q_stand = [1;1;1;1.5;1.5]*pi;
%IK for the robot to find its initial position
IK_robot = @(q) norm(EE_R - (model_R.L*[sin(q) -cos(q)])') + 0.01*norm(q-q_stand);
x0_R = [fmincon(IK_robot,[pi;pi/2;pi;7*pi/4;3*pi/2]);zeros(model_R.nbVarX,1)]; %Find the configuration of the robot to its hand be in EE_R
% The human assistance prediction
[u_H,w,x_H, F,n] = demo_DDP_CP_STS_HUMAN(model,model_H,x0_H);
assistance.u_H = u_H;
assistance.x_H = x_H;
assistance.F = F;
% The robot controller
[u_R,w,x_R,f,n]=demo_DDP_CP_STS_ROBOT_switch(model, model_R, model_H, assistance,x0_R);
end


