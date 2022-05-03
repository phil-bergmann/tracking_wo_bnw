function [Xd Yd]=undistortedToDistortedSensorCoord (Xu, Yu, mKappa1)

% global ifs elses

if ((~Xu && ~Yu) || ~mKappa1)
%     ifs(1)=ifs(1)+1;
    Xd = Xu;
    Yd = Yu;
else %% THIS ONE %%
%     elses(1)=elses(1)+1;
    Ru = sqrt(Xu*Xu + Yu*Yu);
    
    c = 1.0 / mKappa1;
    d = -c * Ru;
    
    Q = c / 3;
    R = -d / 2;
    D = Q*Q*Q + R*R;
    
    if (D >= 0) %% THIS ONE %%
%         ifs(2)=ifs(2)+1;
        %/* one real root */
        D = sqrt(D);
        if (R + D > 0) %% THIS ONE %%
%             ifs(3)=ifs(3)+1;
            S = (R + D)^ (1.0/3.0);
        else
%             elses(3)=elses(3)+1;
            S = -(-R - D)^ (1.0/3.0);
        end
        if (R - D > 0)
%             ifs(4)=ifs(4)+1;
            T = (R - D)^(1.0/3.0);
        else %% THIS ONE %%
%             elses(4)=elses(4)+1;
            T = -(D - R)^(1.0/3.0);
        end
        Rd = S + T;
        
        if (Rd < 0)
            Rd = sqrt(-1.0 / (3 * mKappa1));
            % 				/*fprintf (stderr, "\nWarning: undistorted image point to distorted image point mapping limited by\n");
            % 				fprintf (stderr, "         maximum barrel distortion radius of %lf\n", Rd);
            % 				fprintf (stderr, "         (Xu = %lf, Yu = %lf) -> (Xd = %lf, Yd = %lf)\n\n", Xu, Yu, Xu * Rd / Ru, Yu * Rd / Ru);*/
        end
    else
%         elses(2)=elses(2)+1;
        % 			/* three real roots */
        D = sqrt(-D);
        S = ( sqrt(R*R + D*D))^(1.0/3.0 );
        T = atan2(D, R) / 3;
        sinT = sin(T);
        cosT = cos(T);
        
        % 			/* the larger positive root is    2*S*cos(T)                   */
        % 			/* the smaller positive root is   -S*cos(T) + SQRT(3)*S*sin(T) */
        % 			/* the negative root is           -S*cos(T) - SQRT(3)*S*sin(T) */
        
        Rd = -S * cosT + sqrt(3.0) * S * sinT;	% /* use the smaller positive root */
    end
    
    lambda = Rd / Ru;
    
    Xd = Xu * lambda;
    Yd = Yu * lambda;
end
end