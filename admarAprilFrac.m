
clear
close all
format compact


clear
close all
format compact
set(0,'DefaultFigureWindowStyle','normal')  % 'normal' to un-dock

Dwnsampfac = 3;

hh = newcolormap('turbo');

Sou = 3;   % Starts at 1
Rec = 31;    % Starts at 1  Must be factor of 3, plus 1


validFraction = 0.2;           %0.3;
Nrepdim = 3;                     %3;        % 4
Nhidnodes = 32;                 %128;     % 64
numIterations = 1000;            %750;  % 1500
miniBatchSize = 50;            %512;    % 128  %256


Nclass = 24;   % 38 24

instart = 24;   % 6
inend = 35;    % 24
Nfile = inend - instart + 1;

Prenam = strcat('/Users/nolte/ABCs/Apps/CUSSP/CSouRecApril/CSou',num2str(Sou),'_Rec',num2str(Rec),'_');

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
msk = heaviside0(0.25-mnn);    % Higher nyumbers let more through
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







XValid = XTest;
YValid = YTest;

Iput.NhiddenLayers  =  2;
Iput.Nhidden       =   [128 128];
Iput.Nrepdim       =   3;
Iput.margin       =    1;   % -1 triggers adaptive margin
Iput.numIterations  =  2000;
Iput.miniBatchSize  =  200;
Iput.learningRate  =   1e-4;
Iput.trailingAvg   =   [];
Iput.trailingAvgSq =   [];
Iput.gradDecay    =    0.9;
Iput.gradDecaySq   =   0.99;
Iput.wtinit = 0;

tic
[dlnet,stat] = TripletAMargin(XTrain,YTrain,XValid,YValid,Iput);    % <<<<<<<<<<<<<<<<<<<<<<<<<< TripletAMargin.m
toc

dlXTot = dlarray(single(XTot),'SSCB');
F1Tot = extractdata(predict(dlnet,dlXTot));

Sim = vec2simtest(F1Tot',1);
figure(10)
imagesc(Sim)
%axis equal
colormap(jet)
caxis([0 1])
for floop = 1:Nfile
    line([0 Ntot],[hidate(floop) hidate(floop)],'Color','w','LineWidth',1.5)
    line([hidate(floop) hidate(floop)],[0 Ntot],'Color','w','LineWidth',1.5)
end


dlXTest = dlarray(single(XTest),'SSCB');
F2 = predict(dlnet,dlXTest);
F2e = extractdata(F2);
for cloop = 1:Nclass
    ind = YTest == categorical(cloop);
    F2eav(:,cloop) = mean(F2e(:,ind),2);
end

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
line([RFav(1,1) RFav(1,Nclass)],[RFav(2,1) RFav(2,Nclass)],'Color','k')
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
    distg(cloop) = disgeo/Nclass;
end


Start = F2eav(:,1); End = F2eav(:,Nclass);
vec = (End-Start)/norm(End-Start);   % unit vector along line from start to end

dis = 0; disb = 0; 
for cloop = 2:Nclass
    distmp = dot(F2eav(:,cloop)-F2eav(:,cloop-1),vec);
    dis = dis + distmp;
    dist(cloop) = dis/Nclass;   % Linear distance
    
    distmp = distance(F2eav(:,cloop),F2eav(:,1));
    dista(cloop) = distmp/Nclass;  % absolute distance relative to start

    disb = disb + distance(F2eav(:,cloop-1),F2eav(:,cloop));
    distb(cloop) = disb/Nclass;   % cumulative distance
    
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
xxtmp = 1:numel(dist);
xx = xxtmp*Nfile/Nclass;
%plot(xx,dista,'-o','MarkerFaceColor','r')
plot(xx,dist,'-o',xx,dista,'-o',xx,distb,'-o',xx,distg,'-o')
title('Test Distance')
legend('Linear','Absolute','Accumulated','Geodesic')

Printfile2('distaApril.txt',xx,dista)

