function [u,xc] = cal_u_dynamic(Vx,x,xc)
    load('mat_files/LPV_barycentric_vehicle_reduced_dynamic.mat');
    load('mat_files/controller_matrix.mat');
    load('mat_files/system_matrix.mat');
    if Vx < 6.01
        Vx = 6.01;
    elseif Vx > 35.99
        Vx = 36;
    end
    param = [Vx, 1/Vx];
    n = length(barycentric);
    for i = 1 : n
        vertex_minus_parameter = barycentric(1).vertex(i,:) - param;
        barycentric(i).denominator_of_weight = prod([abs(dot(barycentric(i).rowform_normalvector(1,:), vertex_minus_parameter));
                                                     abs(dot(barycentric(i).rowform_normalvector(2,:), vertex_minus_parameter));]);
        barycentric(i).weight = barycentric(i).numerator_of_weight / barycentric(i).denominator_of_weight;
    end
    total_weight = sum([barycentric(:).weight]);
    for i = 1 : n
        barycentric(i).coordinate = barycentric(i).weight / total_weight;
    end
    F = zeros(size(matrix.F{1}));
    L = zeros(size(matrix.L{1}));
    Q = zeros(size(matrix.Q{1}));
    A = zeros(size(system.A{1}));
    Bu = zeros(size(system.B{1}));
    V = matrix.V;
    U = matrix.U;
    X = matrix.X;
    Y = matrix.Y;
    Cy = eye(4);
    
    for i = 1 : n
        F = F + barycentric(i).coordinate * matrix.F{i};
        L = L + barycentric(i).coordinate * matrix.L{i};
        Q = Q + barycentric(i).coordinate * matrix.Q{i};
        A = A + barycentric(i).coordinate * system.A{i};
        Bu = Bu + barycentric(i).coordinate * system.B{i};
    end
    x = reshape(x,4,1);
    xc = reshape(xc,4,1);
    Ac = inv(V) * (Q - Y * A * X - Y * Bu * L - F * Cy * X)*inv(U);
    Bc = inv(V) * F;
    Cc = L * inv(U);
    disp(Ac);
    disp(Bc);
    u = Cc * xc;
    xc = Ac * xc + Bc * x;    
end