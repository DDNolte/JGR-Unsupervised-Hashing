% FracGroup.m
% Called by:
%   AprilFracGroup.m
%   MayFracGroup.m

clear XTrain YTrain XTest YTest XTot

D = []; DH = []; fraclass = []; last = 0; mnrm = 0; sNdat = 0;
    for floop = 1:Nfile
        indx = (instart-1) + floop;
        load(infilnam{indx})
        [Ndat,~] = size(DHStack);
        sNdat = sNdat + Ndat;
        nrm1 = 1;
        mnrm = mnrm + nrm1;
        DHtmp = DHStack*nrm1;
        lodate(floop) = last + 1;
        hidate(floop) = lodate(floop) + Ndat;
        last = hidate(floop);
        DH = concatmat(DH,DHtmp,1);
        Dtmp = DStack;
        D = concatmat(D,Dtmp,1);
    end
    mnrm = 1;
    D = D/(mnrm/Nfile);
    DH = DH/(mnrm/Nfile);
    

figure(30)
imagesc(DH)
colormap(hh)

nrm = sum(DH(:,150:650),2) + 1e-7;
qq = kron((1./nrm),ones(1,3600));
DHStack = 100*DH.*qq;

[Ntot,~] = size(DHStack);

[lo,hi] = partition_Nbins(Ntot,Nclass);
for cloop = 1:Nclass
    ytrain(lo(cloop):hi(cloop)) = cloop*ones(1,hi(cloop)-lo(cloop)+1);
end


figure(1)
subplot(1,2,1),imagesc(DHStack)
colormap(hh)
title('DHStack')

[Ndat,Ntime] = size(DHStack);

Nfin = Ntime/Dwnsampfac;
[lod,hid] = partition_Nbins(Ntime,Nfin);
clear DSamp
for loop = 1:Nfin
    DSamp(:,loop) = mean(DHStack(:,lod(loop):hid(loop)),2);
end

figure(1)
subplot(1,2,2),imagesc(DSamp)
colormap(hh)
title('DSamp')

DHStack = DSamp;
[Ndat,Ntime] = size(DHStack);


mnn = mean(DHStack);
msk = heaviside0(0.75-mnn);    % 0.25  Higher nyumbers let more through
Msk = kron(ones(Ntot,1),msk);

DHStack = DHStack.*Msk;


valid_frac = 0.4;

ii = 1:Ndat;
ind = randintexc(round(valid_frac*Ndat),Ndat);
NTest = numel(ind);
logicalsum = (ii == ind(1));
for loop = 2:NTest
    logicalsum = logicalsum + (ii == ind(loop));
end


DHSTraintmp = DHStack(not(logical(logicalsum)),:);
DHSTesttmp = DHStack(logical(logicalsum),:);
DHSTottmp = DHStack;

DHSTrain = 10*sqrt(Ntime)*DHSTraintmp/norm(DHSTraintmp);
DHSTest = 10*sqrt(Ntime)*DHSTesttmp/norm(DHSTraintmp);
DHSTot = 10*sqrt(Ntime)*DHSTottmp/norm(DHSTraintmp);

XTrain(1,:,1,:) = DHSTrain';
YTrain = categorical(ytrain(not(logical(logicalsum))));

XTest(1,:,1,:) = DHSTest';
YTest = categorical(ytrain(logical(logicalsum)));

XTot(1,:,1,:) = DHSTot';

NTrain = numel(YTrain);
NTest = numel(YTest);

Iput.validFraction = validFraction;
Iput.Nrepdim = Nrepdim;                     %3;        % 4
Iput.Nhidnodes = Nhidnodes;                 %128;     % 64
Iput.numIterations = numIterations;            %750;  % 1500
Iput.miniBatchSize = miniBatchSize;            %512;    % 128  %256
Iput.NhiddenLayers = 3;
Iput.Nhidden = [128 128 128 128];
Iput.margin = -1;
Iput.learningRate  =   1e-4;
Iput.trailingAvg   =   [];
Iput.trailingAvgSq =   [];
Iput.gradDecay    =    0.9;
Iput.gradDecaySq   =   0.99;
Iput.wtinit = 0;

Iput.Nsplit = 4;



% [lo, hi] = partition_Nbins(NTrain,4);


lasthi = 0;
for loop = 1:Nclass
    n = sum(YTrain==categorical(loop));
    NClassT(loop) = n;
    lo(loop) = lasthi+1;
    hi(loop) = lasthi+n;
    lasthi = hi(loop);
end
for loop = 1:Nclass
    mg(loop,:) = mean(DHSTrain(lo(loop):hi(loop),:));
end


lasthi = 0;
for loop = 1:Nclass
    n = sum(YTest==categorical(loop));
    NClassV(loop) = n;
    lov(loop) = lasthi+1;
    hiv(loop) = lasthi+n;
    lasthi = hiv(loop);
end


figure(2)
plot(mg')
title('Means')

[dlnet,stat,Oput,Stat] = TripletAMargin(XTrain,YTrain,XTest,YTest,Iput);  % <<<<<<<<<<<<<<<< TripletAMargin.m   Main Neural Net Engine

%[Oput,Stat] = aeTrip(Iput,XTrain,YTrain,XTest,YTest);    


Wts1 = Oput.Wts1;
Wtsum = sum(abs(Wts1).^2);

figure(3)
% subplot(2,1,1),pcolor(abs(Wts1).^2),shading interp,colormap(gray),caxis([0 0.001]),title('Input Weights')
% subplot(2,1,2),plot(Wtsum),title('Input Weight Importance')
% axis([0 Ntime 0 max(Wtsum)])


FeatWt = Wtsum/max(Wtsum);
for ploop = 1:NTrain
    FracWt(ploop,:) = DHSTrain(ploop,:).*FeatWt;
end

Amp = sum(FracWt,2);

figure(16)
%plot(Amp)
plot(smooth(Wts1,0.01))
title('Input Weights')
xlabel('Time')

simWt = vec2simtest(FracWt,4);

figure(4)
subplot(1,2,1),imagesc(FracWt),colormap(jet),title('Train Signal*Weight')
subplot(1,2,2),imagesc(simWt),colormap(jet),caxis([-1 1]),title('Sim Matrix of Weighted Signals')

S = blockavg(simWt,50);
figure(15)
plot(S)
xlabel('Time')
ylabel('Self-Similarity')
title('Block Average')


%[X1,X2,pairLabels] = getSiameseBatch(XTrain,YTrain,50);
[X1,X2,X3,simLabel,disLabel] = getTripletBatch(XTrain,YTrain,50);  % <<<<<<<<<<<<<<<<< getTripletBatch

dlX1 = dlarray(single(X2),'SSCB');
% dlX2 = dlarray(single(X3),'SSCB');

dlnet1 = Oput.dlnet1;
% dlnet2 = Oput.dlnet2;

F1 = forward(dlnet1,dlX1);
% F2 = forward(dlnet1,dlX2);


% [Ninternal,~] = size(F1);
% dlX(1,1:Ninternal,1,:) = F1;
% dlX(1,Ninternal+1:2*Ninternal,1,:) = F2;
% dlX = dlarray(dlX,'SSCB');
% 
% F6 = dlarray(forward(dlnet2,dlX),'SB');
% %F6 = dlarray(predict(dlnet2,dlX),'SB');
% 
% hh = newcolormap('turbo');
% figure(5)
% subplot(1,2,1),imagesc(squeeze(X1-X2)),caxis([-0.4 0.4]),colormap(hh),title('Differences')
% mx = max2(abs(extractdata(F6)));
% subplot(1,2,2),imagesc(extractdata(F6)),caxis([-mx mx]),colormap(hh),title('Reconstruction')

dlXTrain = dlarray(single(XTrain),'SSCB');

F1 = predict(dlnet1,dlXTrain);
XF1 = extractdata(F1);
Mdl = fitcecoc(XF1',YTrain);
PredT = predict(Mdl,XF1');

[cm1,cmnrm1] = confusemat(str2num(char(YTrain)),str2num(char(PredT')));

figure(6)
imagesc(cm1);
title('Training Confusion Matrix')
xlabel('Target')
ylabel('Predicted')

figure(7)
bar(cat2num(PredT))
title('Classification of Training Set')

figure(8)
bar3(cm1)
colormap(hh)
title('Confusion Matrix Training')


F1e = extractdata(F1);
for cloop = 1:Nclass
    ind = YTrain == categorical(cloop);
    F1eav(:,cloop) = mean(F1e(:,ind),2);
end

figure(32)
    plot3(F1eav(1,:),F1eav(2,:),F1eav(3,:))

dlXTest = dlarray(single(XTest),'SSCB');
F2 = predict(dlnet1,dlXTest);
F2e = extractdata(F2);
for cloop = 1:Nclass
    ind = YTest == categorical(cloop);
    F2eav(:,cloop) = mean(F2e(:,ind),2);
end

clear fdata
data = F2e';
[xp,yp] = flatten3d(data);
fdata(:,1) = xp; fdata(:,2) = yp;
for cloop = 1:Nclass
    ind = YTest == categorical(cloop);
    Favdata(cloop,:) = mean(fdata(ind,:),1);
end

Start = Favdata(1,:)'; End = Favdata(Nclass,:)';
vec = (End-Start)/norm(End-Start);   % unit vector along line from start to end
theta = acos(vec(1));
R = [cos(theta) -sin(theta);sin(theta) cos(theta)];
Rdata = R*fdata';
xp = Rdata(1,:)';
yp = Rdata(2,:)';
RFav = R*Favdata';

figure(33)
plot(xp,yp,'o')
%line([RFav(1,1) RFav(1,Nclass)],[RFav(2,1) RFav(2,Nclass)],'Color','k')
hold on
plot(RFav(1,:),RFav(2,:),'-or','MarkerFaceColor','r')
plot(RFav(1,1),RFav(2,1),'ok','MarkerSize',12)
hold off
axis equal
title('Flattened')

Start = RFav(:,1); End = RFav(:,Nclass);
vec = (End-Start)/norm(End-Start);   % unit vector along line from start to end
[Y,I]  = max(abs(RFav(2,:)));
devmax = RFav(2,I);
disgeo = 0;
for cloop = 2:Nclass
    pnt = RFav(:,cloop);                         % point on curve
    lindis = dot((pnt-Start),vec)/dot((End-Start),vec);
    if lindis<0
        deriv = 4*devmax;
    elseif lindis > 1
        deriv = -4*devmax;
    else
        deriv = -8*devmax*(lindis-0.5);
    end
    tang = [1,deriv]/norm([1,deriv]);
    
    disgeo = disgeo + dot(RFav(:,cloop)-RFav(:,cloop-1),tang);
    distg(cloop) = disgeo;
end


Start = F2eav(:,1); End = F2eav(:,Nclass);
vec = (End-Start)/norm(End-Start);   % unit vector along line from start to end

dis = 0; disb = 0; 
for cloop = 2:Nclass
    distmp = dot(F2eav(:,cloop)-F2eav(:,cloop-1),vec);
    dis = dis + distmp;
    dist(cloop) = dis;   % Linear distance
    
    distmp = distance(F2eav(:,cloop),F2eav(:,1));
    dista(cloop) = distmp;  % absolute distance relative to start

    disb = disb + distance(F2eav(:,cloop-1),F2eav(:,cloop));
    distb(cloop) = disb;   % cumulative distance
    
%     pnt = F2eav(:,cloop-1);                         % point on curve
%     lindis = dot((pnt-Start),vec)/dot((End-Start),vec);             % linear distance along line
%     devec = (pnt-Start) - lindis*vec;                       % deviation vector from point on line
%     dev = devec/norm(devec+1e-6);                        % magnitude of deviation
%     devmax = norm(devec)/(1-4*(lindis-0.5)^2+1);          % maximum deviation
%     deriv = -8*devmax*(lindis-0.5);
%     
%     localvectmp = (1-deriv)*vec + deriv*dev;
%     localvec = localvectmp/norm(localvectmp);
%     
%     disgeo = disgeo + dot(F2eav(:,cloop)-F2eav(:,cloop-1),localvec);
%     distg(cloop) = disgeo;
%     
    %keyboard
    
end

figure(34)
xx = 1:numel(dist);
%plot(xx,dista,'-o','MarkerFaceColor','r')
plot(xx,dist,'-o',xx,dista,'-o',xx,distb,'-o',xx,distg,'-o')
title('Test Distance')
legend('Linear','Absolute','Accumulated','Geodesic')

disnam = strcat(num2str(Sou),'_',num2str(Rec),'_Dista.txt');
Printfile5(disnam,xx,dist,dista,distb,distg)

% 
% datin(:,1) = xp; datin(:,2) = yp;
% [V,D] = eig(cov(datin));
% datin2 = (datin-mean(datin))*V;
% 
% figure(35)
% plot(datin2(:,1),datin2(:,2),'o')
% 
% xpp = datin2(:,1); ypp = datin2(:,2);
% mat = datin2.^2;
% ovec = mat\ones(size(xpp));
% param = 1./sqrt(ovec);
% 
% par = sqrt(abs(cov(datin2)));
% xx = min(xpp):0.001:max(xpp);
% %yy = param(1)*sqrt(1-xx.^2/param(2));
% yy = par(2,2)*sqrt(1-xx.^2/par(1,1)^2);
% hold on 
% plot(xx,yy,'k')
% plot(xx,-yy,'k')
% hold off
% axis equal
% 

try
    clear Dat gm x y
    for loop = 1:Nclass
        Dat = XF1(:,lo(loop):hi(loop))';
        gm = fitgmdist(Dat,1);
        gmp{loop} = gm;
        x(lo(loop):hi(loop)) = lo(loop):hi(loop);
        y(loop,:) = mvnpdf(XF1',gm.mu,gm.Sigma);
    end
    figure(9)
    semilogy(x,y)
    legend
    title('Train values')
catch
    disp('gm failed')
end


dlXTot = dlarray(single(XTot),'SSCB');
F1Tot = extractdata(predict(dlnet1,dlXTot));

Sim = vec2simtest(F1Tot',1);
figure(10)
imagesc(Sim)
axis equal
colormap(jet)
caxis([0 1])
for floop = 1:Nfile
    line([0 Ntot],[hidate(floop) hidate(floop)],'Color','w','LineWidth',1.5)
    line([hidate(floop) hidate(floop)],[0 Ntot],'Color','w','LineWidth',1.5)

end


Chng = smooth(1-Sim(100,:),0.01);
figure(17)
plot(Chng(20:end-20),'r','LineWidth',1.5)
title('Changes')
ylabel('1-Sim')



dlXTest = dlarray(single(XTest),'SSCB');

F1tst = predict(dlnet1,dlXTest);
XF1tst = extractdata(F1tst);
Pred = predict(Mdl,XF1tst');


[cm2,cmnrm2] = confusemat(str2num(char(YTest)),str2num(char(Pred')));

figure(11)
imagesc(cm2);
title('Validation Confusion Matrix')
xlabel('Target')
ylabel('Label')

figure(12)
bar(cat2num(Pred))
title('Classification of Validation Set')

figure(13)
bar3(cm2)
colormap(hh)
title('Confusion Matrix Validation')

clear Dat gm x y
for loop = 1:Nclass
    x(lov(loop):hiv(loop)) = lov(loop):hiv(loop);
    gm = gmp{loop};
    y(loop,:) = mvnpdf(XF1tst',gm.mu,gm.Sigma);
end



figure(14)
semilogy(x,y)
legend
title('Validation values')


