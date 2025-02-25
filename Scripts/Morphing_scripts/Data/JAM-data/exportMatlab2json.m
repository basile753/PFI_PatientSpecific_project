% [pts,cns] = read_vrml_fast('D:\Queens\Shape\UW\UWModel\ACLC01_R_FemurASC.iv');
[pts,cns] = read_vrml_fast('C:\simtk\uwmodels_2\projects\shape_model\data\mean0\model\Geometry\iv\ACLC_mean_Femur.iv');
cns = cns(:,1:3)+1;
load('D:\Queens\Shape\UW\UWModel\mean\fitpts.mat')

%%
i = 22;
figure(1), clf
plotpatch(pts,cns), hold on
plotpts(pts(fitpts(i).num,:),'r',20), hold off


%%
fitptsjson = fitpts;
for i = 1:22
    if ~isempty(fitptsjson(i).name)
        fitptsjson(i).num = fitptsjson(i).num-1;
    end
end
txt = jsonencode(fitptsjson);

fid = fopen(fullfile('C:\Users\aclouthi\OneDrive - University of Ottawa\Documents\Projects\kneeShape\JAM-data\fitpts.json'),'w');
fprintf(fid,txt);
fclose(fid);

%%
for i = 1:76
    if ~isempty(lig(i).node)
        lig(i).node = lig(i).node - 1;
    end
end
lig(76).name = 'ITB1';
txt = jsonencode(lig);
fid = fopen(fullfile('C:\Users\aclouthi\OneDrive - University of Ottawa\Documents\Projects\kneeShape\JAM-data\ligaments.json'),'w');
fprintf(fid,txt);
fclose(fid);
    
for i = 1:44
    if ~isempty(musc(i).node)
        musc(i).node = musc(i).node - 1;
        musc(i).node(musc(i).node==-1) = NaN;
    end
end
txt = jsonencode(musc);
fid = fopen(fullfile('C:\Users\aclouthi\OneDrive - University of Ottawa\Documents\Projects\kneeShape\JAM-data\muscles.json'),'w');
fprintf(fid,txt);
fclose(fid);
    
%%
[p,~,c] = load_asc('ACLC01-R-tibia-111214.asc','D:\simtk\uwmodels_2\projects\fbknee\model\geometry\ACLC01\');

load('D:\Queens\Shape\UW\UWModel\asc_corresp.mat');

plotpatch(p,c,'b','none',.5), hold on
plotpatch(tib.corresp.vertices,tib.corresp.faces,'r','none',.5), hold off

figure(2), clf
plotpatch(tib.corresp.vertices,tib.corresp.faces,'none','b'), hold on
plotpts(tib.corresp.vertices(lig(end).node(end),:),'r',15)
plotpts(lig(end).pts(3,:)+[.006 0 0],'g',25), hold off