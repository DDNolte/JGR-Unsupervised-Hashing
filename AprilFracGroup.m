% AprilFracGroups.m
% modified from dhtest.m 4/28/24
% Calls:
%       FracGroupHash.m


clear
close all
format compact
set(0,'DefaultFigureWindowStyle','normal')  % 'normal' to un-dock

Dwnsampfac = 3;

hh = newcolormap('turbo');

Sou = 3;   % Starts at 1
Rec = 31;    % Starts at 1  Must be factor of 3, plus 1


% Select pairs of Source and Receivers
% Across:
% 
% 9-1	1-25
% 12-10	4-34x
% 
% 13-1	1-37
% 16-10	4-46x
% 
% 9-13	5-25
% 12-22	8-34
% 
% 13-13	5-37
% 16-22	8-46
% 
% 
% 
% Parallel:
% 
% 9-46	16-25
% 12-37x	13-34
% 
% 1-22	8-1
% 4-13x	5-10

validFraction = 0.2;           %0.3;
Nrepdim = 3;                     %3;        % 4
Nhidnodes = 128;                 %128;     % 64
numIterations = 1500;            %750;  % 1500
miniBatchSize = 50;            %512;    % 128  %256


Nclass = 24;

instart = 24;
inend = 35;
Nfile = inend - instart + 1;

Prenam = strcat('CSou/CSou',num2str(Sou),'_Rec',num2str(Rec),'_');

infilnam{1} = strcat(Prenam,'20220318');
infilnam{2} = strcat(Prenam,'20220319');
infilnam{3} = strcat(Prenam,'20220320');
infilnam{4} = strcat(Prenam,'20220321');
infilnam{5} = strcat(Prenam,'20220322');
infilnam{6} = strcat(Prenam,'20220323');
infilnam{7} = strcat(Prenam,'20220324');
infilnam{8} = strcat(Prenam,'20220325');
infilnam{9} = strcat(Prenam,'20220326');
infilnam{10} = strcat(Prenam,'20220327');
infilnam{11} = strcat(Prenam,'20220328');
infilnam{12} = strcat(Prenam,'20220329');
infilnam{13} = strcat(Prenam,'20220330');
infilnam{14} = strcat(Prenam,'20220331');
infilnam{15} = strcat(Prenam,'20220401');
infilnam{16} = strcat(Prenam,'20220402');
infilnam{17} = strcat(Prenam,'20220403');
infilnam{18} = strcat(Prenam,'20220404');
infilnam{19} = strcat(Prenam,'20220405');
infilnam{20} = strcat(Prenam,'20220406');
infilnam{21} = strcat(Prenam,'20220407');
infilnam{22} = strcat(Prenam,'20220408');
infilnam{23} = strcat(Prenam,'20220409');
infilnam{24} = strcat(Prenam,'20220410');
infilnam{25} = strcat(Prenam,'20220411');
infilnam{26} = strcat(Prenam,'20220412');
infilnam{27} = strcat(Prenam,'20220413');
infilnam{28} = strcat(Prenam,'20220414');
infilnam{29} = strcat(Prenam,'20220415');
infilnam{30} = strcat(Prenam,'20220416');
infilnam{31} = strcat(Prenam,'20220417');
infilnam{32} = strcat(Prenam,'20220418');
infilnam{33} = strcat(Prenam,'20220419');
infilnam{34} = strcat(Prenam,'20220420');
infilnam{35} = strcat(Prenam,'20220421');
infilnam{36} = strcat(Prenam,'20220422');
infilnam{37} = strcat(Prenam,'20220423');
infilnam{38} = strcat(Prenam,'20220424');
infilnam{39} = strcat(Prenam,'20220427');
infilnam{40} = strcat(Prenam,'20220428');
infilnam{41} = strcat(Prenam,'20220429');
infilnam{42} = strcat(Prenam,'20220430');

FracGroupHash


