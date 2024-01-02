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

x1 = zeros(300,20);
x2 = zeros(300,20);
x3 = zeros(300,20);
x4 = zeros(300,20);
x5 = zeros(300,20);
dimension = 512;
Cov = zeros(dimension,dimension);
mean = zeros(1,dimension);
npoints = 300;      %number of points in each class;
index=0;
newdimension=20;
X = zeros(1500,20);
for j=1:5
    Cov=zeros(512,512);
    index=(j-1)*300;
    x2 = Xtrain(index+1:index+300,:);
    for k=1:npoints
        mean = mean + x2(k,:);
    end
    mean = mean/npoints;
   for i=1:npoints
      Cov = Cov + (x2(i,:)-mean(1,:))' * (x2(i,:)-mean(1,:));
   end
   Cov=Cov/npoints;
   
   [V,D] = eig(Cov);
   [B,id] = sort(diag(D),'descend');
   V = V(:,id); 
   EigVec = zeros(dimension,newdimension);
   EigVec(:,1:newdimension) = V(:,1:newdimension);
   x1=Xtrain(index+1:index+300,:) * EigVec;
   X(index+1:index+300,:)=x1;
end

Xnew=zeros(1150,newdimension);
Xnew(1:230,:) = X(1:230,:);
Xnew(231:460,:) = X(301:530,:);
Xnew(461:690,:) = X(601:830,:);
Xnew(691:920,:) = X(901:1130,:);
Xnew(921:1150,:) = X(1201:1430,:);

Ynew=zeros(1150,1);
Ynew(1:230,1) = 1;
Ynew(231:460,1) = 2;
Ynew(461:690,1) = 3;
Ynew(691:920,1) = 4;
Ynew(921:1150,1) = 5;

%initialisation of weights and biases
n0=newdimension;
n1=5;
n2=5;
n3=5;
w11=rand(n1,n0);
w22=rand(n2,n1);
w33=rand(n3,n2);

b11=rand(n1,1);
b22=rand(n2,1);
b33=rand(n3,1);