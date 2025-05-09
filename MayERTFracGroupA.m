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
numIterations = 500;            %750;  % 1500
miniBatchSize = 100;            %512;    % 128  %256


Nclass = 37;

instart = 15;
inend = 35;
Nfile = inend - instart + 1;

Prenam = strcat('CSouERT/CSou',num2str(Sou),'_Rec',num2str(Rec),'_');

infilnam{1} = strcat(Prenam,'20220501');
infilnam{2} = strcat(Prenam,'20220502');
infilnam{3} = strcat(Prenam,'20220503');
infilnam{4} = strcat(Prenam,'20220504');
infilnam{5} = strcat(Prenam,'20220505');
infilnam{6} = strcat(Prenam,'20220506');
infilnam{7} = strcat(Prenam,'20220507');
infilnam{8} = strcat(Prenam,'20220508');
infilnam{9} = strcat(Prenam,'20220509');
infilnam{10} = strcat(Prenam,'20220510');
infilnam{11} = strcat(Prenam,'20220511');
infilnam{12} = strcat(Prenam,'20220512');
infilnam{13} = strcat(Prenam,'20220513');
infilnam{14} = strcat(Prenam,'20220514');
infilnam{15} = strcat(Prenam,'20220515');
infilnam{16} = strcat(Prenam,'20220516');
infilnam{17} = strcat(Prenam,'20220517');
infilnam{18} = strcat(Prenam,'20220518');

infilnam{19} = strcat(Prenam,'20220519');
infilnam{20} = strcat(Prenam,'20220520');
infilnam{21} = strcat(Prenam,'20220521');
infilnam{22} = strcat(Prenam,'20220522');
infilnam{23} = strcat(Prenam,'20220523');
infilnam{24} = strcat(Prenam,'20220524');
infilnam{25} = strcat(Prenam,'20220525');
infilnam{26} = strcat(Prenam,'20220526');
infilnam{27} = strcat(Prenam,'20220527');
infilnam{28} = strcat(Prenam,'20220528');
infilnam{29} = strcat(Prenam,'20220529');
infilnam{30} = strcat(Prenam,'20220530');
infilnam{31} = strcat(Prenam,'20220531');

infilnam{32} = strcat(Prenam,'20220601');
infilnam{33} = strcat(Prenam,'20220602');
infilnam{34} = strcat(Prenam,'20220603');
infilnam{35} = strcat(Prenam,'20220604');

AFracGroupHash


