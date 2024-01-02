function [] = Plot2(A, B, C)         
    
    
    dt = delaunayTriangulation(A,B);
    tri = dt.ConnectivityList;
    Ai = dt.Points(:,1);
    Bi = dt.Points(:,2);
    
    F = scatteredInterpolant(A,B,C);
    
    Ci = F(Ai,Bi);
    
    trisurf(tri,Ai,Bi,Ci)    
    shading interp
end