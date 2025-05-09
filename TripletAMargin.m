% dlnet = TripletNN(XTrain,YTrain,XValid,YValid,Iput)
% siamdr.m
% https://www.mathworks.com/help/deeplearning/ug/train-a-siamese-network-for-dimensionality-reduction.html
% https://www.mathworks.com/help/deeplearning/examples.html
%
% XTrain(1,Nbm,1,NTrain)
% YTrain(1:NTrain)
% XValid(1,Nbm,1,NValid)
% YValid(1:NValid)
%
% Iput.NhiddenLayers    2
% Iput.Nhidden          [20 10]
% Iput.Nrepdim          3
% Iput.margin           1
% Iput.numIterations    750
% Iput.miniBatchSize    200
% Iput.learningRate     1e-4
% Iput.trailingAvg      []
% Iput.trailingAvgSq    []
% Iput.gradDecay        0.9
% Iput.gradDecaySq      0.99
% Iput.wtinit


function [dlnet,stat,Oput,Stat] = TripletAMargin(XTrain,YTrain,XValid,YValid,Iput)

wrt = 0;

[Nchan1,Nchan2,~,Nsamp] = size(XTrain);
dlXValid = dlarray(single(XValid),'SSCB');
dlXTrain = dlarray(single(XTrain),'SSCB');

NhiddenLayers = Iput.NhiddenLayers;
Nhidden = Iput.Nhidden;
Nrepdim = Iput.Nrepdim;
if Iput.margin > 0
    margin = Iput.margin;
    adapt = 0;
else
    adapt = 1;
    idxclass = unique(YTrain);
    Nclass = numel(idxclass);
    for cloop = 1:Nclass
        idx = YTrain == idxclass(cloop);
        Tmp = mean(squeeze(XTrain(1,:,1,idx)),2);
        ClassMean(cloop,:) = Tmp;
    end
    
    for yloop = 1:Nclass
        for xloop = 1:Nclass
            Dmargin(yloop,xloop) = 1 + sqrt(sum((ClassMean(yloop,:) - ClassMean(xloop,:)).^2));
        end
    end
    
    figure(103)
    imagesc(Dmargin)
    colormap(jet)
    colorbar
    title('Dmargin: Distance')
    
end

numIterations = Iput.numIterations;  % 750
miniBatchSize = Iput.miniBatchSize;

learningRate = Iput.learningRate;
trailingAvg = Iput.trailingAvg;
trailingAvgSq = Iput.trailingAvgSq;
gradDecay = Iput.gradDecay;
gradDecaySq = Iput.gradDecaySq;

% Define Network Architecture
if NhiddenLayers == 1
    layers = [
        imageInputLayer([Nchan1 Nchan2],'Name','input1','Normalization','none')
        weightLayer;
        fullyConnectedLayer(Nhidden(1),'Name','fc1','WeightsInitializer','he')
        reluLayer('Name','relu1')
        fullyConnectedLayer(Nrepdim,'Name','fc5','WeightsInitializer','he')];
elseif NhiddenLayers == 2
    layers = [
        imageInputLayer([Nchan1 Nchan2],'Name','input1','Normalization','none')
        weightLayer;
        fullyConnectedLayer(Nhidden(1),'Name','fc1','WeightsInitializer','he')
        reluLayer('Name','relu1')
        fullyConnectedLayer(Nhidden(2),'Name','fc4','WeightsInitializer','he')
        reluLayer('Name','relu2')
        fullyConnectedLayer(Nrepdim,'Name','fc5','WeightsInitializer','he')];
elseif NhiddenLayers == 3
    layers = [
        imageInputLayer([Nchan1 Nchan2],'Name','input1','Normalization','none')
        weightLayer;
        fullyConnectedLayer(Nhidden(1),'Name','fc1','WeightsInitializer','he')
        reluLayer('Name','relu1')
        fullyConnectedLayer(Nhidden(2),'Name','fc2','WeightsInitializer','he')
        reluLayer('Name','relu2')
        fullyConnectedLayer(Nhidden(3),'Name','fc3','WeightsInitializer','he')
        reluLayer('Name','relu3')
        fullyConnectedLayer(Nrepdim,'Name','fc5','WeightsInitializer','he')];
elseif NhiddenLayers == 4
    layers = [
        imageInputLayer([Nchan1 Nchan2],'Name','input1','Normalization','none')
        weightLayer;
        fullyConnectedLayer(Nhidden(1),'Name','fc1','WeightsInitializer','he')
        reluLayer('Name','relu1')
        fullyConnectedLayer(Nhidden(2),'Name','fc2','WeightsInitializer','he')
        reluLayer('Name','relu2')
        fullyConnectedLayer(Nhidden(3),'Name','fc3','WeightsInitializer','he')
        reluLayer('Name','relu3')
        fullyConnectedLayer(Nhidden(4),'Name','fc4','WeightsInitializer','he')
        reluLayer('Name','relu4')
        fullyConnectedLayer(Nrepdim,'Name','fc5','WeightsInitializer','he')];
else
    disp('Wrong HniddenLayers in TripletNN')
end

lgraph = layerGraph(layers);

dlnet = dlnetwork(lgraph);

if Iput.wtinit > 0
    
    [~,~,~,Wtinit,~] = WBClass(X2FV(XTrain),YTrain);
    
    AA = dlnet.Learnables.Value{1};
    [nrow,~] = size(AA);
    for loop = 1:nrow
        %         BB(loop,:) = AA(loop,:).*sqrt(Wtinit);
        BB(loop,:) = AA(loop,:).*Wtinit.^(Iput.wtinit);
    end
    dlnet.Learnables.Value{1} = BB;
    
end

%analyzeNetwork(dlnet)


% Define Model Gradients Function and Specify Training Options

executionEnvironment = "auto";

plots = "training-progress";

plotRatio = 16/9;

if plots == "training-progress"
    trainingPlot = figure(731);
    trainingPlot.Position(3) = plotRatio*trainingPlot.Position(4);
    trainingPlot.Visible = 'on';
    trainingPlot.Position = [158 439 510 424];
    
    trainingPlotAxes = gca;
    trainingPlotAxes.YScale = 'log';
    
    lineLossTrain = animatedline(trainingPlotAxes);
    xlabel(trainingPlotAxes,"Iteration")
    ylabel(trainingPlotAxes,"Loss")
    title(trainingPlotAxes,"Loss During Training")
    
    lineAccPlot = figure(730);
    AccPlotAxes = gca;
    AccPlotAxes.YScale = 'log';
    
    lineAccPlot.Visible = 'on';
    lineAccTrain = animatedline(AccPlotAxes,'Color','r');
    xlabel(AccPlotAxes,"Iteration")
    ylabel(AccPlotAxes,"Accuracy")
    title(AccPlotAxes,"Accuracy During Training")
    lineAccValid = animatedline(AccPlotAxes,'Color','b');
    
end

dimensionPlot = figure(732);
dimensionPlot.Position(3) = plotRatio*dimensionPlot.Position(4);
dimensionPlot.Visible = 'on';
dimensionPlot.Position = [670 438 518 424];

dimensionPlotAxes = gca;

uniqueGroups = unique(YTrain);
colors = hsv(length(uniqueGroups));

% Train Model
% Loop over mini-batches.
frame_count = 0; indup = 0;
for iteration = 1:numIterations
    
    % Extract mini-batch of image triplets and triplet labels
    [X1,X2,X3,simLabel,disLabel] = getTripletBatch(XTrain,YTrain,miniBatchSize);
    
    if adapt == 1   % Adaptive margins
        for mloop = 1:miniBatchSize
            margin(mloop) = Dmargin(simLabel(mloop),disLabel(mloop));
        end
    end
    
    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % 'SSCB' (spatial, spatial, channel, batch) for image data
    dlX1 = dlarray(single(X1),'SSCB');
    dlX2 = dlarray(single(X2),'SSCB');
    dlX3 = dlarray(single(X3),'SSCB');
    
    % If training on a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlX1 = gpuArray(dlX1);
        dlX2 = gpuArray(dlX2);
        dlX3 = gpuArray(dlX3);
    end
    
    % Evaluate the model gradients and the generator state using
    % dlfeval and the modelGradients function listed at the end of the
    % example.
    [gradients,loss] = dlfeval(@TmodelGrad,dlnet,dlX1,dlX2,dlX3,margin);    % <<<<<<<<<<<<<<<<<<<< TmodelGrad at bottom of this file
    lossValue = double(gather(extractdata(loss)));
    
    % Update the Siamese network parameters.
    [dlnet.Learnables,trailingAvg,trailingAvgSq] = ...
        adamupdate(dlnet.Learnables,gradients, ...
        trailingAvg,trailingAvgSq,iteration,learningRate,gradDecay,gradDecaySq);
    
    if mod(iteration,25)==0
        indup = indup + 1;
        
        % Update the training loss progress plot.
        if plots == "training-progress"
            figure(trainingPlot);
            addpoints(lineLossTrain,iteration,lossValue);
            drawnow
            %axis([0 numIterations 0.1 100])
            if (mod(iteration,5)==0)&&(wrt==1)
                frame_count = frame_count + 1;
                print('-dpng',strcat('Converg/tt',num2str(frame_count)));
            end
        end
        
        
        % Update the reduced-feature plot of the test data.
        % Compute reduced features of the test data:
        dlFValid = predict(dlnet,dlXValid);
        FValid = extractdata(dlFValid);
        %         dlFValid = predict(dlnet,dlXTrain);
        %         FValid = extractdata(dlFValid);
        
        
        if Nrepdim > 2
            figure(dimensionPlot);
            for k = 1:length(uniqueGroups)
                % Get indices of each image in test data with the same numeric
                % label (defined by the unique group):
                ind = YValid==uniqueGroups(k);
                % Plot this group:
                plot3(dimensionPlotAxes,gather(FValid(1,ind)'),gather(FValid(2,ind)'),gather(FValid(3,ind)'),'.','color',...
                    colors(k,:),'MarkerSize',14);
                grid on
                %axis([-3.5 3.5 -3.5 3.5])
                hold on
            end
            
            %legend(uniqueGroups)
            
            % Update title of reduced-feature plot with training progress information.
            title(dimensionPlotAxes,"3-D Latent-Space Representation. Iteration = " +...
                iteration);
            legend(dimensionPlotAxes,'Location','eastoutside');
            xlabel(dimensionPlotAxes,"Feature 1")
            ylabel(dimensionPlotAxes,"Feature 2")
            
            hold off
            drawnow
        end
        
        dlXTrain = dlarray(single(XTrain),'SSCB');
        FXTrain = extractdata(forward(dlnet,dlXTrain));
        
        Mdl = fitcecoc(FXTrain',YTrain);
        
        PredTrain = predict(Mdl,FXTrain');
        AccuTrain = sum(rowvec(PredTrain)==rowvec(YTrain));
        AccuracyTrain = AccuTrain/numel(YTrain);
        
        dlXValid = dlarray(single(XValid),'SSCB');
        FXValid = extractdata(forward(dlnet,dlXValid));
        
        Pred = predict(Mdl,FXValid');
        AccuTest = sum(rowvec(Pred)==rowvec(YValid));
        
        Accuracy = AccuTest/numel(YValid);
        
        if plots == "training-progress"
            figure(lineAccPlot);
            addpoints(lineAccValid,iteration,Accuracy);
            addpoints(lineAccTrain,iteration,AccuracyTrain);
            drawnow
        end
        
        AccuV(indup) = Accuracy;
        AccuT(indup) = AccuracyTrain;
        
    end     % end if mod
    
    
    if (mod(iteration,5)==0)&&(wrt==1)
        print('-dpng',strcat('Latent/tt',num2str(frame_count)))
    end
    
end  % end iteration

stat.Accuracy = AccuV;
stat.AccuracyTrain = AccuT;

Oput.XTrain = XTrain;
Oput.YTrain = YTrain;
Oput.dlnet1 = dlnet;

Wts1 = extractdata(dlnet.Learnables.Value{1});
Oput.Wts1 = Wts1;

Stat.Nrepdim = Nrepdim;
Stat.numIterations = numIterations;
Stat.miniBatchSize = miniBatchSize;


    function [gradients, loss] = TmodelGrad(net,X1,X2,X3,margin)
        % The modelGradients function calculates the contrastive loss between the
        % paired images and returns the loss and the gradients of the loss with
        % respect to the network learnable parameters
        
        % Pass anchor image forward through the network
        F1 = forward(net,X1);
        % Pass similar image  forward through the network
        F2 = forward(net,X2);
        % Pass dissimilar image forward through the network
        F3 = forward(net,X3);
        
        % Calculate Triplet contrastive loss
        closs = TcontrastiveLoss(F1,F2,F3,margin);  % Triplet loss
        
        % Regularization: Calculate the elastic sum of first-layer weights
        lambda = 100;
        expon = 0.5;
        q = net.Learnables.Value{1};
        qelastic = (sum(sum(abs(q).^expon))/numel(q)).^(1/expon);
        regloss = lambda*qelastic;
        
        loss = closs + regloss;
        
        % Calculate gradients of the loss with respect to the network learnable
        % parameters
        gradients = dlgradient(loss, net.Learnables);
        
        %     displine('closs = ',closs)
        %     displine('regloss = ',regloss)
        
        if regloss > 10*closs
            displine('From modelGradients: regloss > closs',regloss/closs)
        end
        
    end



end

