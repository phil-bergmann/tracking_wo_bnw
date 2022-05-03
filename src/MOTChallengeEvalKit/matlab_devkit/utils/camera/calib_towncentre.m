function Frames=calib_towncenter(Frames,namefile)

fid=fopen(namefile);
C=textscan(fid,'%s');
c=C{1};

fx=str2double(cell2mat(c(3)));
fy=str2double(cell2mat(c(6)));
px=str2double(cell2mat(c(9)));
py=str2double(cell2mat(c(12)));
sk=str2double(cell2mat(c(15)));
tx=str2double(cell2mat(c(18)));
ty=str2double(cell2mat(c(21)));
tz=str2double(cell2mat(c(24)));
rx=str2double(cell2mat(c(27)));
ry=str2double(cell2mat(c(30)));
rz=str2double(cell2mat(c(33)));
rw=str2double(cell2mat(c(36)));
k1=str2double(cell2mat(c(39)));
k2=str2double(cell2mat(c(42)));
p1=str2double(cell2mat(c(45)));
p2=str2double(cell2mat(c(48)));

R=[1-2*ry^2-2*rz^2,2*rx*ry-2*rz*rw,2*rx*rz+2*ry*rw;2*rx*ry+2*rz*rw,1-2*rx^2-2*rz^2,2*ry*rz-2*rx*rw;2*rx*rz-2*ry*rw,2*ry*rz+2*rx*rw,1-2*rx^2-2*ry^2];

R=[R,[tx;ty;tz]];

K=[fx,0,px;0,fy,py;0,0,1];

P=K*R;
P(:,3)=[];

R(:,3)=[];

for fr=1:numel(Frames)
    for p=1:numel(Frames(fr).id)
    
            u=double(Frames(fr).ximg(p));
            v=double(Frames(fr).yimg(p));
            
            x2=(u-px)/fx;
            y2=(v-py)/fy;
            
            Pu=undistort(k1,k2,p1,p2,[x2;y2]);
            xu=Pu(1);
            yu=Pu(2);

            Pd2=homotrans(inv(R),[Pu;1]);
            
           % Pd=homotrans(inv(P),[u;v;1]);
            
            Frames(fr).x(p)=Pd2(1);
            Frames(fr).y(p)=Pd2(2);
            Frames(fr).z(p)=0;
            Frames(fr).vx(p)=0;
            Frames(fr).vy(p)=0;
            Frames(fr).vz(p)=0;
            Frames(fr).ximg(p)=round(Frames(fr).ximg(p));
            Frames(fr).yimg(p)=round(Frames(fr).yimg(p));
    end
end
    

end

function x=undistort(k1,k2,p1,p2,xd)

    xd=double(xd);
    x = xd;                             % initial guess
    
    for kk=1:20,
        
        r_2 = sum(x.^2);
        k_radial =  double(1 + k1 * r_2 + k2 * r_2.^2 );
        delta_x = double([2*p1*x(1,:).*x(2,:) + p2*(r_2 + 2*x(1,:).^2);
        p1 * (r_2 + 2*x(2,:).^2)+2*p2*x(1,:).*x(2,:)]);
        x = (xd - delta_x)./(ones(2,1)*k_radial);
            
    end;
    
end

function t = homotrans(P,v)

[dim,npts] = size(v);

if ~all(size(P)==dim)
    error('Transformation matrix and point dimensions do not match');
end

t = P*v;  % Transform

for r = 1:dim-1     %  Now normalise
    t(r,:) = t(r,:)./t(end,:);
end

t(end,:) = ones(1,npts);

end
