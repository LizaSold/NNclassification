function NN

global colors markertypes;

clc;
clear all;

colors = {[1 0 0], [0 1 0], [ 0 0 1], [1 1 0], [1 0 1], [0 1 1], 1/255*[255 165 0], 1/255*[47 79 79], 1/255*[188 143 143], 1/255*[139 69 19], 1/255*[0 191 255] };  % ����� ��� ������� ������
markertypes = { 'x', 'o', '*', 'diamond','x', 'o', '*', 'diamond', 'x', 'o', '*', 'diamond' };    % ���� �������� ��� ������� ������

NtrainAll = 15000;   % ����� ��������� �������
NtestAll = 5000;   % ����� �������� �������
[PtrainC, TtrainC, PtestC, TtestC] = ex8_convert(NtrainAll, NtestAll);

% ��������, ��������, ������������� ��
NN_hidden_neurons = [150];       % ���������� �������� � ������� �����
type_train_func = 4;    % ������� �������� (1-traingd, 2-traingda, 3-traingdm, 4-traingdx, 5-trainrp, 6-traincgf, 7-traincgb, 8-traincgp, 9-trainscg, 10-trainlm, 11-trainbfg, 12-trainoss, 13-trainbr)
type_perform_fcn = 2;   % �������������� ������� (1-mae, 2-mse, 3-sae, 4-sse, 5-crossentropy)
NN_arch_type = 1;   % ����������� (1 - ������� �� ������� ���������������, 2 - ��������� �� ��)
NN_out_type = 1;    % �� ��������� ���� (1 - purelin, 2-tansig, 3-logsig, 4-satlin, 5 - softmax), softmax ���������� ������������ � ��������� � crossentropy

% �������� �� ������� ���������������
net = create_neural_network(NN_arch_type, NN_out_type, NN_hidden_neurons);
net = init(net);        % ������������� ������� ������������� � ��������

net = SetTrainParam(net, type_train_func, type_perform_fcn);
net = train_neural_network(net, PtrainC, TtrainC);
ytrain = sim_neural_network(net, PtrainC);  % ����� �� �� ��������� �������
ytest = sim_neural_network(net, PtestC);    % ����� �� �� �������� �������

[~,ytrain_c] = max(ytrain); [~,ytest_c] = max(ytest);
[e_mean, e1, e2, c] = calc_errors(ytest_c, TtestC);

%����� ��������
tr = csvread ( 'train.csv' , 1, 0);% ������ train.csv
colormap(gray)
for i = 1:NtrainAll*2
if(i==1||i==NtrainAll+1)
  figure;
end

if(i<NtrainAll+1&&i<26)
  subplot(5,5,i) 
  elseif(i<NtrainAll+26&&i>NtrainAll)
    k1=i-NtrainAll
    subplot(5,5,i-NtrainAll)  
end

digit = reshape(tr(i, 2:end), [28,28])';    % row = 28 x 28 image
imagesc(digit)

if(i<NtrainAll+1&&i<26)
  title(num2str(ytrain_c(i)),'Color', 'r')
  elseif(i<NtrainAll+26&&i>NtrainAll)
     k2=i-NtrainAll
     title(num2str(ytest_c(i-NtrainAll+1)),'Color', 'r')
  end
end
end

%% ������ ������
function [e_mean, e1, e2, c] = calc_errors(ytest_c, TtestC)
e_mean = sum(sum(ytest_c ~= TtestC)) / length(TtestC)
c = confusionmat(TtestC',ytest_c')
c1 = c - diag(diag(c));
e1 = sum(c1)./sum(c)
e2 = sum(c1')./sum(c')
end

%% ������� ������
function [PtrainC, TtrainC, PtestC, TtestC] = ex8_convert(NtrainAll, NtestAll)
global data_type axis_rect C;
% ��������� ���������� ����������� �������
% PtestC - ������� ������� �������� ��� �������� ������� �����������
% ����� �������� �������� * ����� ��������� (2)
% PtrainC - ������� ������� �������� ��� ��������� ������� �����������
% ����� ��������� �������� * ����� ��������� (2)
% TtestC - ������� �������� �������� �������� ��� �������� ������� �����������
% 1 * ����� �������� ��������
% TtrainC - ������� �������� �������� �������� ��� ��������� ������� �����������
% 1 * ����� ��������� ��������
% TtrainC(1,i)=j ��������, ��� i-� ��������� ������ ��������� � ������ j
% ��������� ������
% data_type - ����� ����� ������� (�.�. ������ 11)
% C(datatype) - ����� �������
% axis_rect(datatype) - ���������� ������� ������������� ������� �������� �
% ������� [xmin xmax ymin ymax]

data_type = 12;
C(data_type) = 10;
% axis_rect{data_type} = [0 28 0 28];

%������ �������� � ��������� �������� ��� 10 ������� (10 ����)
%������ ����� ��� ������� �����
%� ����� https://www.kaggle.com/c/digit-recognizer
%The dataset is made available
%under a Creative Commons Attribution-Share Alike 3.0 license

%������ ������� - �������� �����
%2-784 ������� - ������� ����������� 28x28
%�.�. �������� ��� ������� � ������������ ������ ��������
%���� �������� ����� �� ���-�� ��������
end_digits=NtrainAll;

tr = csvread ( 'train.csv' , 1, 0);% ������ train.csv 

n = size(tr, 1); % ���������� �������� � dataset

%��������� 1� �������, � ������� ���������� �������� ��������
targets  = tr(1:end_digits,1); 
targets(targets == 0) = 10;% ������ ����� '0' �� ������ '10'

%�������� ������� ����������� ��������� �������� �� �������
targetsd = dummyvar(targets);

inputs = tr(1:end_digits,2:end); % the rest of columns are predictors

%������������� ���������� �������
inputs = inputs';%100 �������� �� 784 �����
targets = targets';%100 �������� �� 1 ������
targetsd = targetsd';%100 �������� �� 10 �����

PtrainC=inputs.';
PtestC=tr(end_digits:end_digits*2,2:end);

TtrainC=targets;
TtestC=tr(end_digits:end_digits*2,1);
end

%% NN - �������
% �������� ��
function net = create_neural_network(NN_arch_type, NN_out_type, NN_hidden_neurons)
net = [];
if NN_arch_type == 1
    % ff
    net = feedforwardnet(NN_hidden_neurons);
elseif NN_arch_type == 2
    net = cascadeforwardnet(NN_hidden_neurons);
    %cascade
end;
if NN_out_type == 1
    % sigmoid
    net.layers{length(net.layers)}.transferFcn = 'purelin';
elseif NN_out_type == 2
    % sigmoid
    net.layers{length(net.layers)}.transferFcn = 'logsig';
elseif NN_out_type == 3
    % sigmoid
    net.layers{length(net.layers)}.transferFcn = 'tansig';
elseif NN_out_type == 4
    % sigmoid
    net.layers{length(net.layers)}.transferFcn = 'satlin';
elseif NN_out_type==5
    net.layers{net.numLayers}.transferFcn = 'softmax';
    % softmax
end;
end

%% �������� ��
function net = train_neural_network(net, dataTrain, groupsTrain)
C = max(groupsTrain);
NS_out = zeros(C,length(groupsTrain));

for i = 1:C
    NS_out(i,:) = (groupsTrain == i);
end;
% dataTrain
% length(dataTrain)
% size(dataTrain,1)
% size(dataTrain,2)
net = train(net, dataTrain', NS_out);
end

%% ������� ���������� ��
function net = SetTrainParam(net, type_train_func, type_perform_fcn)
trainf_fcns = {'traingd','traingda', 'traingdm', 'traingdx', 'trainrp'... % 1-5
    'traincgf', 'traincgb', 'traincgp', 'trainscg',... % 6-9
    'trainlm', 'trainbfg', 'trainoss', 'trainbr' };   % 10-13
perform_fcns = {'mae', 'mse', 'sae', 'sse', 'crossentropy'};
net.trainfcn = trainf_fcns{type_train_func};        % ������� ��������
net.performfcn = perform_fcns{type_perform_fcn};    % ������� ���������� ������


if type_perform_fcn == 5
    net.performParam.regularization = 0.1;
    net.performParam.normalization = 'none';
    net.outputs{length(net.layers)}.processParams{2}.ymin = 0;
end;

trainParam = net.trainParam;

trainParam.epochs = 1000;           % ������������ �������� ����� ���� ��������
trainParam.time = Inf;              % ������������ ����� ��������
trainParam.goal = 0;                % ������� �������� ������
trainParam.min_grad = 1e-07;        % �������� ��������� ��� ��������
trainParam.max_fail = 32;            % ������������ ����� ���� ��� ������� ��������

% ��������� ������������ ��������
trainParam.showWindow = true;       % ���������� ���� ��� ���
trainParam.showCommandLine = false; % �������� � ��������� ������ ��� ���
trainParam.show = 25;               % ������� ���������� - ����� ������� ����

switch type_train_func
    case 1
        % traingd - ����������� �����
        trainParam.lr = 0.01;               % !�������� ��������
    case 2
        % traingda - ����������� ����� c ����������
        trainParam.lr = 0.01;               % !�������� �������� (�����������)
        trainParam.lr_inc = 1.05;           % !����������� ���������� �������� ��������
        trainParam.lr_dec = 0.7;            % !����������� ���������� �������� ��������
        trainParam.max_perf_inc  = 1.04;    % !���������� ����������� ��������� ������
        % ��� ��� ���������� �������� ����������� � lr_dec ���, ������������ �� ����������, � ��������� ������ ������������ ����������
        % ���� ������� ������ ������ ����������, �� �������� ������������� � lr_inc ���
    case 3
        % traingm - ����������� ����� c ����������
        trainParam.lr = 0.01;               % !�������� ��������
        trainParam.mc = 0.9;                % !������ ������� (�� 0 �� 1), ��� �� ������ ��� ����� ������� ��������� �������������
        % ��� mc=0 traingdm ��������� � traingd
    case 4
        % traingx - ����������� ����� c ���������� � ��������
        trainParam.lr = 0.01;               % !�������� �������� (�����������)
        trainParam.mc = 0.9;                % !������ �������
        trainParam.lr_inc = 1.05;           % !����������� ���������� �������� ��������
        trainParam.lr_dec = 0.7;            % !����������� ���������� �������� ��������
        trainParam.max_perf_inc  = 1.04;    % !���������� ����������� ��������� ������
    case 5
        % trainrp
        trainParam.lr = 0.01;               % !�������� �������� (�����������)
        % ��������� ��������� (��� ������ �������� delta)
        trainParam.delt_inc = 1.2;          % Increment to weight change
        trainParam.delt_dec = 0.5;          % Decrement to weight change
        trainParam.delta0 = 0.07;           % Initial weight change
        trainParam.deltamax = 50.0;         % Maximum weight change
    case {6,7,8, 11, 12}
        % traincgf, traincgp, traincgb, trainbfg, trainoss
        if ~isempty(find(type_train_func == [6 7 8], 1))
            trainParam.searchFcn = 'srchcha';   % !������� ����������� ��������� ������ (srchbac, srchbre, srchgol, srchhyb)
        else
            trainParam.searchFcn = 'srchbac';   % !������� ����������� ��������� ������ (srchbac, srchbre, srchgol, srchhyb)
        end
        % ��������� ������� ����������� ������
        trainParam.scale_tol = 20;         % Divide into delta to determine tolerance for linear search.
        trainParam.alpha = 0.001;           % Scale factor that determines sufficient reduction in perf
        trainParam.beta = 0.1;              % Scale factor that determines sufficiently large step size
        trainParam.delta = 0.01;            % Initial step size in interval location step
        trainParam.gama = 0.1;              % Parameter to avoid small reductions in performance, usually set to 0.1 (see srch_cha)
        trainParam.low_lim = 0.1;           % Lower limit on change in step size
        trainParam.up_lim = 0.5             % Upper limit on change in step size
        trainParam.max_step = 100;           % Maximum step length
        trainParam.min_step = 1.0e-6;        % Minimum step length
        trainParam.bmax = 26;               % Maximum step size
        if type_train_func == 11
            trainParam.batch_frag = 0;          % In case of multiple batches, they are considered independent. Any nonzero value implies a fragmented batch, so the final layer's conditions of a previous trained epoch are used as initial conditions for the next epoch.
        end;
    case 9
        % trainscgf
        trainParam.sigma = 5e-5;            % ��������� ����� ��� ������������� ������ �����������
        trainParam.lambda = 5e-7;           % �������� ��� ������������� ��� ������ �������������� ������� �����
        %-------------------------------------------------------%
        %-------������ ���������� �������-----------------------%
        %-------------------------------------------------------%
    case {10, 13}
        % trainlm, trainbr
        % ��������� ��������� (��� ������ �������� mu)
        trainParam.mu = 0.001;              % Initial mu
        trainParam.mu_dec = 0.1;            % mu decrease factor
        trainParam.mu_inc = 10;             % mu increase factor
        trainParam.mu_max = 1e10;           % Maximum mu
end;
net.trainParam = trainParam;
end

%% ������������� ��
function y = sim_neural_network(net, dataAll)
y = sim(net, dataAll');
end
