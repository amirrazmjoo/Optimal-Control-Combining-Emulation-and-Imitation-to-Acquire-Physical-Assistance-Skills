function h = plotCoM(x, sz, col)

if nargin<2
	sz = .05;
end
if nargin<3
	col = [0 0 0];
end

nbPts = 40;
t = linspace(0, 2*pi, nbPts);
tl = [1:nbPts*.25, nbPts*.75:-1:nbPts*.5];
h = [];
for n=1:size(x,2)
	msh = sz * [cos(t); sin(t)] + repmat(x(:,n), 1, nbPts);
	h = [h, plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',col)];
	h = [h, patch(msh(1,tl), msh(2,tl), col,'linestyle','none')];
end