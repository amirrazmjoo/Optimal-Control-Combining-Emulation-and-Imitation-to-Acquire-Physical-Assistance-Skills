function h = plotPanda(a, pos0, lnw, edgecolor, facecolor, alpha)
% Display of a planar robot arm
%
% Sylvain Calinon, 2020

if nargin<2
	pos0 = [0; 0; -4];
end
if nargin<3
	lnw = 3;
end
if nargin<4
	edgecolor = [0,0,0];
end
if nargin<5
	facecolor = [1,1,1];
end
if nargin<6
	alpha = 1;
end
if size(pos0,1)==2
	pos0 = [pos0; -4];
end

% load('data/PandaMsh.mat');
load('data/PandaMsh2.mat');
% clr = lines(4) * .8;
% plot(0, 0, 'r+');

%BASE
p = pos0(1:2) + [0; 81];
for i=1:length(msh{1})
	x = msh{1}{i} + repmat(p, 1, size(msh{1}{i},2));
	if i==1
		h = patch(x(1,:), x(2,:), pos0(3)+ones(1,size(x,2))*3, facecolor,'linewidth',lnw,'edgecolor',edgecolor);
	else
		h = [h, plot3(x(1,:), x(2,:), pos0(3)+ones(1,size(x,2))*3, '-','linewidth',lnw,'color',edgecolor)];
	end
end

%LINK 1
R = [cos(a(1)), -sin(a(1)); sin(a(1)), cos(a(1))];
for i=1:length(msh{2})
	x = R * msh{2}{i} + repmat(p, 1, size(msh{2}{i},2));
	if i==1
		h = [h, patch(x(1,:), x(2,:), pos0(3)+ones(1,size(x,2))*2, facecolor,'linewidth',lnw,'edgecolor',edgecolor)];
	else
		h = [h, plot3(x(1,:), x(2,:), pos0(3)+ones(1,size(x,2))*2, '-','linewidth',lnw,'color',edgecolor)];
	end
end

%LINK 2
p = p + R * [0; 79]; 
aTmp = sum(a(1:2));
R = [cos(aTmp), -sin(aTmp); sin(aTmp), cos(aTmp)];
for i=1:length(msh{3})
	x = R * msh{3}{i} + repmat(p, 1, size(msh{3}{i},2));
	if i==1
		h = [h, patch(x(1,:), x(2,:), pos0(3)+ones(1,size(x,2))*0, facecolor,'linewidth',lnw,'edgecolor',edgecolor)];
	else
		h = [h, plot3(x(1,:), x(2,:), pos0(3)+ones(1,size(x,2))*0, '-','linewidth',lnw,'color',edgecolor)];
	end
end

%LINK 3
p = p + R * [0; 96];
aTmp = sum(a(1:3));
R = [cos(aTmp), -sin(aTmp); sin(aTmp), cos(aTmp)];
for i=1:length(msh{4})
	x = R * msh{4}{i} + repmat(p, 1, size(msh{4}{i},2));
	if i<6
		h = [h, patch(x(1,:), x(2,:), pos0(3)+ones(1,size(x,2))*1+.1*i, facecolor,'linewidth',lnw,'edgecolor',edgecolor)];
	else
		h = [h, plot3(x(1,:), x(2,:), pos0(3)+ones(1,size(x,2))*1+.1*i, '-','linewidth',lnw,'color',edgecolor)];
	end
end

%END-EFFECTOR
p = p + R * [-20.5; 51];
plot(p(1), p(2), 'k.');