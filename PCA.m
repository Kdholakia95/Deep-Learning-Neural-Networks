imcar = readNPY('car.npy');
imcat = readNPY('cat.npy');
imdog = readNPY('dog.npy');
imship = readNPY('ship.npy');
imtruck = readNPY('truck.npy');

Xtrain = zeros(1500,512);
Xtrain(1:300,:) = imcar;
Xtrain(301:600,:) = imcat;
Xtrain(601:900,:) = imdog;
Xtrain(901:1200,:) = imship;
Xtrain(1201:1500,:) = imtruck;

Ytrain=zeros(1500,1);
Ytrain(1:300,1) = 1;
Ytrain(301:600,1) = 2;
Ytrain(601:900,1) = 3;
Ytrain(901:1200,1) = 4;
Ytrain(1201:1500,1) = 5;

%-------------------------apply PCA----------------------------
npoints=1500;
dimension = 512;
Cov = zeros(dimension,dimension);
mean = zeros(1,dimension);

for i=1:npoints
    mean = mean + Xtrain(i,:);
end  
mean = mean/npoints;

for i=1:npoints
    Cov = Cov + (Xtrain(i,:)-mean(1,:))' * (Xtrain(i,:)-mean(1,:));
end
Cov=Cov/npoints;

newdimension = 512;
[V,D] = eig(Cov);
[B,id] = sort(diag(D),'descend');
V = V(:,id);
EigVec = zeros(dimension,newdimension);
EigVec(:,1:newdimension) = V(:,1:newdimension); %Extracting top 20 eig vectors

XtrainNew = Xtrain * EigVec;   % projecting the data on to eigen vectors
YtrainNew = Ytrain;      % y reamins same as previous one

Xnew=zeros(1150,newdimension);
Xnew(1:230,:) = XtrainNew(1:230,:);
Xnew(231:460,:) = XtrainNew(301:530,:);
Xnew(461:690,:) = XtrainNew(601:830,:);
Xnew(691:920,:) = XtrainNew(901:1130,:);
Xnew(921:1150,:) = XtrainNew(1201:1430,:);

Ynew=zeros(1150,1);
Ynew(1:230,1) = 1;
Ynew(231:460,1) = 2;
Ynew(461:690,1) = 3;
Ynew(691:920,1) = 4;
Ynew(921:1150,1) = 5;