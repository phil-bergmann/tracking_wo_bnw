function camPar=parseCameraParameters(camconffile)
%
%
% (C) Anton Andriyenko, 2012
%
% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without written permission from the authors.

% if static cam
% if isempty(strfind(camconffile,'%'))

[pt fl ex]=fileparts(camconffile);
if isequal(ex,'.mat')
    load(camconffile);
    return;
else
    xDoc=xmlread(fullfile(camconffile));
    
     camname=xDoc.getElementsByTagName('Camera').item(0).getAttribute('name');
    if strcmpi(camname,'PRML_LAB')
        camPar.ortho=1; % Bird's eye view orthographic
        camPar.mR = str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('scale'));
        camPar.scale = camPar.mR;
        % fill in rest
        camPar.mT=0;
        camPar.mInt=0;
        camPar.mGeo=0;
    else
        camPar.ortho=0; % Bird's eye view orthographic
        width=str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('width'));
        height=str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('height'));
        
        ncx=str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('ncx'));
        nfx=str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('nfx'));
        dx=str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('dx'));
        dy=str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('dy'));
        dpx=str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('dpx'));
        dpy=str2double(xDoc.getElementsByTagName('Geometry').item(0).getAttribute('dpy'));
        
        focal=str2double(xDoc.getElementsByTagName('Intrinsic').item(0).getAttribute('focal'));
        kappa1=str2double(xDoc.getElementsByTagName('Intrinsic').item(0).getAttribute('kappa1'));
        cx=str2double(xDoc.getElementsByTagName('Intrinsic').item(0).getAttribute('cx'));
        cy=str2double(xDoc.getElementsByTagName('Intrinsic').item(0).getAttribute('cy'));
        sx=str2double(xDoc.getElementsByTagName('Intrinsic').item(0).getAttribute('sx'));
        
        tx=str2double(xDoc.getElementsByTagName('Extrinsic').item(0).getAttribute('tx'));
        ty=str2double(xDoc.getElementsByTagName('Extrinsic').item(0).getAttribute('ty'));
        tz=str2double(xDoc.getElementsByTagName('Extrinsic').item(0).getAttribute('tz'));
        rx=str2double(xDoc.getElementsByTagName('Extrinsic').item(0).getAttribute('rx'));
        ry=str2double(xDoc.getElementsByTagName('Extrinsic').item(0).getAttribute('ry'));
        rz=str2double(xDoc.getElementsByTagName('Extrinsic').item(0).getAttribute('rz'));
        
        mGeo.mImgWidth = width;
        mGeo.mImgHeight = height;
        mGeo.mNcx = ncx;
        mGeo.mNfx = nfx;
        mGeo.mDx = dx;
        mGeo.mDy = dy;
        mGeo.mDpx = dpx;
        mGeo.mDpy = dpy;
        
        
        %% intrinsic
        mInt.mFocal = focal;
        mInt.mKappa1 = kappa1;
        mInt.mCx = cx;
        mInt.mCy = cy;
        mInt.mSx = sx;
        
        %% extrinsic
        mExt.mTx = tx;
        mExt.mTy = ty;
        mExt.mTz = tz;
        mExt.mRx = rx;
        mExt.mRy = ry;
        mExt.mRz = rz;
        
        %% inverted
        mT=[tx;ty;tz];
        sa = sin(rx);
        ca = cos(rx);
        sb = sin(ry);
        cb = cos(ry);
        sg = sin(rz);
        cg = cos(rz);
        
        mR11 = cb * cg;
        mR12 = cg * sa * sb - ca * sg;
        mR13 = sa * sg + ca * cg * sb;
        mR21 = cb * sg;
        mR22 = sa * sb * sg + ca * cg;
        mR23 = ca * sb * sg - cg * sa;
        mR31 = -sb;
        mR32 = cb * sa;
        mR33 = ca * cb;
        
        mR=[mR11 mR12 mR13;
            mR21 mR22 mR23;
            mR31 mR32 mR33];
        
        transform=[mR mT;[0 0 0 1]];
        invtrans=inv(transform);
        tx=invtrans(1,4);ty=invtrans(2,4);tz=invtrans(3,4);
        
        mExt.mTxi = tx;
        mExt.mTyi = ty;
        mExt.mTzi = tz;
        
        camPar.mExt=mExt;
        camPar.mInt=mInt;
        camPar.mGeo=mGeo;
        
        %% mkappa<0
        if kappa1<0
            warning('CAREFUL! kappa1 < 0, cam derivates are wrong!');
        end
        
        [camPar.mR camPar.mT]=getRotTrans(camPar);
        % else
        %     global sceneInfo
        %     F=length(sceneInfo.frameNums);
        %     for t=1:F
        %
        %     end
        % end
        % end
    end
end