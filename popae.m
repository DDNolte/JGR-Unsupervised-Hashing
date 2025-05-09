%popae
% Calls:
%   siaMae.m
%   Calls:
%       ModelGradients2.m

clear
clear global
format compact
close all
set(0,'DefaultFigureWindowStyle','docked')  % 'normal' to un-dock

%global aviobj

hh = newcolormap('heated');

%filnam = 'Ch5Sflip_40_4class';  % Re-analysis of April 24 with sqrt norm from sixfracprep4class.m
filnam = 'Ch5S4classRe424';  % Re-analysis of April 24 (train-validation) with better first-arrival alignment sixfracprep4class.m
%filnam = 'Ch5Gflip_40_4class';  % Re-analysis of April 17 with sqrt norm from sixfracprep4class.m
%filnam = 'Ch5GSeq-42-417';   % April 17
%filnam = 'Ch5GSeqJ10-42';   % New July 10 sequential data

load(filnam)

[sy0,sx0] = size(ZeroFrac);
[sy1,sx1] = size(OneFrac);
[sy2,sx2] = size(TwoFrac);
[sy3,sx3] = size(TwoFrac2);

s(1) = mean(std(ZeroFrac(:,4:sx0),[],2));  
s(2) = mean(std(OneFrac(:,4:sx1),[],2));
s(3) = mean(std(TwoFrac(:,4:sx2),[],2));
s(4) = mean(std(TwoFrac2(:,4:sx3),[],2));
scale_fac = 1/mean(s);

symin = min([sy0,sy1,sy2,sy3]);
symax = max([sy0,sy1,sy2,sy3]);
ind0 = randint(symax,sy0);
ind1 = randint(symax,sy1);
ind2 = randint(symax,sy2);
ind3 = randint(symax,sy3);

sy0 = symax;
sy1 = symax;
sy2 = symax;
sy3 = symax;

Npat = 4*symax;

ZeroFractmp = ZeroFrac(ind0,:);
OneFractmp = OneFrac(ind1,:);
TwoFractmp = TwoFrac(ind2,:);
TwoFrac2tmp = TwoFrac2(ind3,:);

ZeroFracp = scale_fac*ZeroFractmp(:,4:sx0);
OneFracp = scale_fac*OneFractmp(:,4:sx0);
TwoFracp = scale_fac*TwoFractmp(:,4:sx0);
TwoFrac2p = scale_fac*TwoFrac2tmp(:,4:sx0);
TestSetp = scale_fac*TestSet(:,4:sx0);
Yfraclass = TestSet(:,2);
TestSet = TestSetp;

m(1,:) = mean(ZeroFracp);
m(2,:) = mean(OneFracp);
m(3,:) = mean(TwoFracp);
m(4,:) = mean(TwoFrac2p);

figure(1)
for loop = 1:4
   plot(m(:,1:800)')
end


Frac = concatmat(ZeroFracp,[],1);
Frac = concatmat(Frac,OneFracp,1);
Frac = concatmat(Frac,TwoFracp,1);
Frac = concatmat(Frac,TwoFrac2p,1);

fraclass(1:sy0) = ones(1,sy0);
fraclass(sy0+1:sy0+sy1) = 2*ones(1,sy1);
fraclass(sy0+sy1+1:sy0+sy1+sy2) = 3*ones(1,sy2);
fraclass(sy0+sy1+sy2+1:sy0+sy1+sy2+sy3) = 4*ones(1,sy3);

Nclass = 4;
NpatC(1) = sy0;
NpatC(2) = sy1;
NpatC(3) = sy2;
NpatC(4) = sy3;

minsize = min(NpatC);

Ntrain = sum(NpatC);


%%  Downsample Data

[Npat,sx] = size(Frac);         % number of patterns in training set
[Ntest,~] = size(TestSet);      % number of patterns in test set

Filterf = 0.1;      % for low-pass filter on timeseries data
TargSize = 601;    % down-sampling size
FVncol = 200;       % number of features


for loop = 1:Npat
   Fracf(loop,:) = lpfilter(Frac(loop,:),Filterf);
end
Targ = zeros(Npat,TargSize);
Fracg = congrid(Fracf,Targ);    % Fracg is downsampled timeseries data (training set)

for loop = 1:Ntest
   TestSetf(loop,:) = lpfilter(TestSet(loop,:),Filterf);
end
Targ = zeros(Ntest,TargSize);
TestSetg = congrid(TestSetf,Targ);  % TestSetg is downsamples timeseries data (TestSet)

figure(200)
imagesc(Frac)   % original data
colormap(jet)
caxis([-4 4])
axis([1 sx 1 Ntrain])
title('Train Frac')

depth = Frac(:,3);
[Y,I] = sort(depth,'descend');
Fracsort = Frac(I,:);

figure(201)
imagesc(Fracg)  % down-sampled data
colormap(jet)
caxis([-1 1])
axis([1 FVncol 1 Ntrain])
title('Fracg')

figure(202)
imagesc(TestSetg)
colormap(jet)
caxis([-1 1])
axis([1 FVncol 1 Ntest])
title('Test')

MeanFracg = mean(Fracg);
MeanTestSetg = mean(TestSetg);

t = 1:FVncol;
figure(203)
plot(t,MeanFracg(1:FVncol),t,MeanTestSetg(1:FVncol))
title('Average Signals')
legend('Train','Validation')

mg(1,:) = mean(Fracg(1:symax,1:FVncol));
mg(2,:) = mean(Fracg(symax+1:2*symax,1:FVncol));
mg(3,:) = mean(Fracg(2*symax+1:3*symax,1:FVncol));
mg(4,:) = mean(Fracg(3*symax+1:4*symax,1:FVncol));

figure(204)
plot(mg')
legend



%%

XTrain(1,:,1,:) = Fracg(:,1:FVncol)';
%XTrain = dlarray(XTrain,'SSCB');
YTrain = categorical(fraclass);
YTest = categorical(Yfraclass);


validFraction = 0.3;
Nrepdim = 3;        % 4
Nhidnodes = 128;     % 64
numIterations = 1000;  % 1500
miniBatchSize = 1024;
wtinit = 1;

split_training = 1;  %1 = split total   2 = split each class
Nsplit = 20;

if split_training == 1
   [YTrain,fraclassplit,Nclass] = spliRtrain(fraclass,Nsplit); 
   [YTestmp,~,~] = spliRtrain(Yfraclass,Nsplit); 
   YTest = YTestmp';
elseif split_training == 2
   [YTrain,fraclassplit,Nclass] = splitrain(fraclass,Nsplit); 
   [YTestmp,~,~] = splitrain(Yfraclass,Nsplit); 
   YTest = YTestmp';
end

%     moviename = 'UnsuperVisedMov';
%     aviobj = VideoWriter(moviename,'MPEG-4');
%     aviobj.FrameRate = 8;
%     open(aviobj);

[dlnet1,dlnet2] = siaMae(XTrain,YTrain,Nrepdim,Nhidnodes,validFraction,numIterations,miniBatchSize,wtinit);  % Training function call

Wts1 = extractdata(dlnet1.Learnables.Value{1});
Wtsum = sum(abs(Wts1).^2);

figure(4)
subplot(2,1,1),imagesc((Wts1)),colormap(hh),caxis([-0.3 0.3]),title('Input Weights')
subplot(2,1,2),plot(Wtsum),title('Input Weight Importance') 

[X1,X2,pairLabels] = getSiameseBatch(XTrain,YTrain,180);

dlX1 = dlarray(single(X1),'SSCB');
dlX2 = dlarray(single(X2),'SSCB');

F1 = forward(dlnet1,dlX1);
F2 = forward(dlnet1,dlX2);


[Ninternal,~] = size(F1);
dlX(1,1:Ninternal,1,:) = F1;
dlX(1,Ninternal+1:2*Ninternal,1,:) = F2;
dlX = dlarray(dlX,'SSCB');

F6 = dlarray(forward(dlnet2,dlX),'SB');
%F6 = dlarray(predict(dlnet2,dlX),'SB');


figure(5)
subplot(1,2,1),imagesc(squeeze(X1-X2));colormap(hh);caxis([-1 1]),title('Differences')
subplot(1,2,2),imagesc(extractdata(F6));colormap(hh);caxis([-1 1]),title('Reconstruction')

ChanImportance = sum(abs(extractdata(F6)).^2,2);

xx = 1:FVncol;
Printfile7('popout.txt',xx,mg(1,:),mg(2,:),mg(3,:),mg(4,:),ChanImportance',Wtsum);

[sigW, sigB, lam, stdratio, stdratioC] = WBClass(mg,[1 2 3 4]);
WBImportance = 0.375./stdratio;

XX = Fracg(:,1:200);
[sigW, sigB, lam, stdratio, stdratioC] = WBClass(XX,fraclassplit);
WBImportance2 = stdratio;

figure(6)
plot(xx,ChanImportance/max(ChanImportance),xx,WBImportance2/max(WBImportance2))
title('Channel Importance')
legend('Recon Importance','WB')

figure(7)
plot(xx,Wtsum/max(Wtsum),xx,WBImportance2/max(WBImportance2))
title('Channel Importance')
legend('InputWeight','WBImport')

dlXTrain = dlarray(single(XTrain),'SSCB');

F1 = forward(dlnet1,dlXTrain);
XF1 = extractdata(F1);
Mdl = fitcecoc(XF1',YTrain);
Pred = predict(Mdl,XF1');

figure(8)
plotconfusion(YTrain',Pred);
title('Training')

[cm,cmnrm] = confusemat(str2num(char(YTrain)),str2num(char(Pred)));

figure(9)
bar(cat2num(Pred))
title('Classification of Training Set')

figure(10)
bar3(cmnrm)
colormap(hh)
title('Normalized Confusion Matrix Training')

XTest(1,:,1,:) = TestSetg(:,1:FVncol)';
dlXTest = dlarray(single(XTest),'SSCB');

F2 = forward(dlnet1,dlXTest);
XF2 = extractdata(F2);
PredTest = predict(Mdl,XF2'); 

figure(11)
bar(cat2num(PredTest))
title('Classification of Test Set')


figure(12)
plotconfusion(YTest,PredTest);
title('Confusion Matrix Test Set ECOC')

[cm2,cmnrm2] = confusemat(str2num(char(YTest)),str2num(char(PredTest)));

figure(13)
bar3(cmnrm2)
colormap(hh)
title('Normalized Confusion Matrix on Test from ECOC')

% trainClassifier was run on 4-dim data
if Nrepdim == 4
    XDataLearn = XF1;
    XDataLearn(5,:) = YTrain;
    
    [trainedClassifier, validationAccuracy] = trainClassifierWKNN(XDataLearn);
    yfit = trainedClassifier.predictFcn(XF2);
    validationAccuracy
    
    figure(31)
    plotconfusion(YTest,categorical(yfit));
    title('WKNN Confusion Matrix on Test Set')
    
    [cm3,cmnrm3] = confusemat(str2num(char(YTest)),yfit);
    
    fh = figure(13);
    fh.Position = [306 1167 1413 310];
    dum = set(fh);
    
    subplot(1,3,1),bar3(cmnrm3),colormap(hh),title('Normalized Confusion Matrix WKNN');
    subplot(1,3,2),imagesc(cmnrm3),colormap(hh),colorbar,title('Normalized Confusion Matrix');
    subplot(1,3,3),bar(cmnrm3'),legend,title('Normalized Confusion Matrix WKNN'),xlabel('Target Classes');
    set(gcf,'Color','white')
        
    figure(32)
    bar(yfit)
    title('Classification of Test Set WKNN')
    
    
end

figure(15)
imagesc(XF2)
colormap(jet)
caxis([-0.5 0.5])
title('Test Internal Representation')
NTsum = 0;
for loop = 1:Nclass
   NT(loop) = sum(YTest == categorical(loop)); 
   NTsum = NTsum + NT(loop);
   line([NTsum+0.5 NTsum+0.5], [Nrepdim+0.5 0],'LineWidth',3)
end

tsne_flag = 0;
if tsne_flag == 1
    figure(16)
    YTSNE = tsne(XF2','Algorithm','exact','Distance','minkowski');
    HH = gscatter(YTSNE(:,1),YTSNE(:,2),YTest,'','',[12 12 12 12]);
    title('Test T-SNE')
end

simXF1 = vec2simtest(XF1',1);
simXF2 = vec2simtest(XF2',1);

figure(17)
subplot(1,2,1),imagesc(simXF1),colormap(hh),caxis([0 1]),title('Train Internal-Rep Sim Matrix')
NTsum = 0;
for loop = 1:Nclass
   NTr(loop) = sum(YTrain == categorical(loop)); 
   NTsum = NTsum + NTr(loop);
   line([NTsum+0.5 NTsum+0.5], [Ntrain+0.5 0],'LineWidth',2,'Color','k')
   line([Ntrain+0.5 0],[NTsum+0.5 NTsum+0.5] ,'LineWidth',2,'Color','k')
end
subplot(1,2,2),imagesc(simXF2),colormap(hh),caxis([0 1]),title('Test Internal-Rep Sim Matrix')
NTsum = 0;
for loop = 1:Nclass
   NTe(loop) = sum(YTest == categorical(loop)); 
   NTsum = NTsum + NTe(loop);
   line([NTsum+0.5 NTsum+0.5], [Ntest+0.5 0],'LineWidth',2,'Color','k')
   line([Ntest+0.5 0],[NTsum+0.5 NTsum+0.5] ,'LineWidth',2,'Color','k')
end


CL = 0;
if CL == 1
	classificationLearner
end

Dat1 = XF1(:,1:symax);
gm1 = fitgmdist(Dat1',1);
Dat2 = XF1(:,symax+1:2*symax);
gm2 = fitgmdist(Dat2',1);
Dat3 = XF1(:,2*symax+1:3*symax);
gm3 = fitgmdist(Dat3',1);
Dat4 = XF1(:,3*symax+1:4*symax);
gm4 = fitgmdist(Dat4',1);

x = 1:4*symax;
y(1,:) = mvnpdf(XF1',gm1.mu,gm1.Sigma);
y(2,:) = mvnpdf(XF1',gm2.mu,gm2.Sigma);
y(3,:) = mvnpdf(XF1',gm3.mu,gm3.Sigma);
y(4,:) = mvnpdf(XF1',gm4.mu,gm4.Sigma);

figure(18)
semilogy(x,y(1,:),x,y(2,:),x,y(3,:),x,y(4,:))
legend
title('Train values')

FeatWt = Wtsum/max(Wtsum);
for ploop = 1:Npat
    FracWt(ploop,:) = Fracg(ploop,1:FVncol).*FeatWt;
end

simWt = vec2simtest(FracWt,4);

figure(19)
subplot(1,2,1),imagesc(FracWt),colormap(jet),title('Train Signal*Weight')
subplot(1,2,2),imagesc(simWt),colormap(jet),caxis([-1 1]),title('Sim Matrix of Weighted Signals')


for ploop = 1:Ntest
    TestFracWt(ploop,:) = TestSetg(ploop,1:FVncol).*FeatWt;
end

simTWt = vec2simtest(TestFracWt,4);

figure(20)
subplot(1,2,1),imagesc(TestFracWt),colormap(jet),title('Test Signal*Weight')
subplot(1,2,2),imagesc(simTWt),colormap(jet),caxis([-1 1]),title('Sim Matrix of Weighted Signals')


% k-means back to original class number

F1idx = kmeans(XF1',4);
F2idx = kmeans(XF2',4);

cmF1 = confusemat(fraclass,F1idx);
cmF2 = confusemat(Yfraclass,F2idx);

figure(31)
subplot(1,2,1),bar3(cmF1)
subplot(1,2,2),bar3(cmF2)
colormap(hh)

b1 = sort(mat2vec(cmF1),'ascend');
b2 = sort(mat2vec(cmF2),'ascend');

M = numel(b1);
N = sqrt(M);

acc1 = 1 - sum(b1(1:M-N))/sum(b1)
acc2 = 1 - sum(b2(1:M-N))/sum(b2)







