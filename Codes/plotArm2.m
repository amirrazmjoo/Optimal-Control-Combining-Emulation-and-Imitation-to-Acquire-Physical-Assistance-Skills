function h = plotArm2(a, d, p, sz,ft_l,ft_h,dir,facecolor, edgecolor, alpha)
% Display of a planar robot arm.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
% }
%
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.


if nargin<4
	sz = .05;
end
if nargin<8
	facecolor = [.5,.5,.5];
end
if nargin<9
	edgecolor = [.99,.99,.99];
end
if nargin<10
	alpha = 1;
end
if size(p,1)==2
	p = [p; 1];
end
nbSegm = 30;
t = linspace(pi/2,0,nbSegm);
% t2 = linspace(pi,0,nbSegm/2);
Feat(1,:) = [-sz -sz+ft_l-ft_h -sz+ft_l-ft_h+ft_h*cos(t) -sz]*dir;
Feat(2,:) = [ft_h ft_h ft_h*sin(t) 0]-0.08;
Feat(3,:) = zeros(1,length(Feat(1,:)));
p_Feat = p + Feat;
h = patch(p_Feat(1,:),p_Feat(2,:),repmat(p(3),1,length(Feat(1,:))),facecolor,'edgeColor',edgecolor,'linewidth',3,'edgealpha',alpha,'facealpha',alpha); 
   
% h = plotArmBasis(p, sz, facecolor, edgecolor, alpha);
% h=[];
for i=1:length(a)
	[p, hTmp] = plotArmLink2(a(i), d(i), p+[0;0;.1], sz, facecolor, edgecolor, alpha);
	h = [h hTmp];
end
