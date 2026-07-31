function make_figures()
% MAKE_FIGURES  Reproduce all PICUP-FDD manuscript figures in MATLAB.
%
% HOW TO RUN
%   1. Put this file in ...\PICUP-FDD\final_figures\
%   2. In MATLAB, cd to that folder and type:  make_figures
%   3. PNG + PDF (300 dpi, vector where possible) are written next to this file.
%
% EXACT DATA SOURCE PER FIGURE  (all paths relative to ..\results\)
%   Figure 1  overconfidence concept ...... none (illustrative curves)
%   Figure 2  unified architecture ........ none (schematic)
%   Figure 3  class balance + baselines ... tep_class_distribution.csv
%                                           baselines_metrics.csv
%                                           deep_baselines_metrics.csv
%   Figure 4  variable saliency ........... unified_interpretability.csv
%   Figure 5  calibrated operating point .. unified_detection.csv
%   Figure 6  open-set AUROC by score ..... unified_openset.csv
%
% The RICHER versions of Fig 5 (full FAR-FDR curve) and Fig 6 (score
% distributions + ROC) need the raw per-window score arrays. Run step09 on a
% machine with PyTorch; it dumps arr_*.npy. Export those to CSV
% (arr_fault_score.csv, arr_scores.csv) and the two optional sections at the
% bottom will draw the curve/distribution versions automatically.
% -------------------------------------------------------------------------

here    = fileparts(mfilename('fullpath'));
resDir  = fullfile(here, '..', 'results');
outDir  = here;

% ---- shared styling ----
BLUE=[0.173 0.435 0.733]; RED=[0.753 0.224 0.169]; GREEN=[0.153 0.682 0.376];
GREY=[0.498 0.549 0.553]; YELLOW=[0.976 0.906 0.624]; PURPLE=[0.843 0.741 0.851];
ORANGE=[0.902 0.494 0.133]; SALMON=[0.961 0.718 0.694]; LIGHTBLUE=[0.839 0.918 0.973];
set(0,'DefaultAxesFontName','Helvetica','DefaultAxesFontSize',11, ...
      'DefaultAxesBox','off','DefaultAxesLineWidth',0.9);

save_fig = @(f,name) local_save(f, outDir, name);

%% ===================== FIGURE 1 — overconfidence concept ================
f = figure('Color','w','Position',[100 100 900 360]);
x = linspace(0,1,100);
subplot(1,2,1);
y1 = 0.5 + 0.45*tanh(x*3);
area(x,y1,'FaceColor',BLUE,'FaceAlpha',0.10,'EdgeColor',BLUE,'LineWidth',2.2); hold on;
patch([0.6 1 1 0.6],[0 0 1 1],RED,'FaceAlpha',0.07,'EdgeColor','none');
ylim([0 1]); xlabel('Distance from training distribution');
ylabel('Reported confidence'); title('Softmax classifier','FontWeight','bold');
text(0.8,0.32,{'confident','AND wrong'},'Color',RED,'HorizontalAlignment','center', ...
     'FontWeight','bold','FontSize',9);
subplot(1,2,2);
y2 = 0.1 + 0.85*x.^1.6;
area(x,y2,'FaceColor',GREEN,'FaceAlpha',0.10,'EdgeColor',GREEN,'LineWidth',2.2); hold on;
patch([0.6 1 1 0.6],[0 0 1 1],GREEN,'FaceAlpha',0.07,'EdgeColor','none');
ylim([0 1]); xlabel('Distance from training distribution');
ylabel('Reported uncertainty'); title('Uncertainty-aware model','FontWeight','bold');
text(0.8,0.32,{'flags the','unknown'},'Color',GREEN,'HorizontalAlignment','center', ...
     'FontWeight','bold','FontSize',9);
sgtitle('Figure 1.  A confident classifier is not a trustworthy one','FontWeight','bold','FontSize',13);
save_fig(f,'fig1_overconfidence');

%% ===================== FIGURE 2 — architecture schematic ================
f = figure('Color','w','Position',[100 100 1100 430]); ax=axes(f); axis(ax,'off');
xlim(ax,[0 11]); ylim(ax,[0 4.4]); hold(ax,'on');
drawbox = @(x,y,w,h,t,c) local_box(ax,x,y,w,h,t,c);
drawarr = @(x1,y1,x2,y2) annotation('arrow'); % placeholder; real arrows below
drawbox(0.2,1.7,1.5,1.0,{'Windowed','input','(W x d)'},LIGHTBLUE);
drawbox(2.1,1.7,2.0,1.0,{'Conv-BN-ReLU','-Pool'},BLUE.*0+[0.68 0.84 0.90]);
drawbox(4.5,1.7,2.0,1.0,{'Conv-ReLU','-GAP'},[0.68 0.84 0.90]);
drawbox(6.9,1.7,1.7,1.0,{'128-d','features'},YELLOW);
local_flowarrow(ax,1.7,2.2,2.1,2.2);
local_flowarrow(ax,4.1,2.2,4.5,2.2);
local_flowarrow(ax,6.5,2.2,6.9,2.2);
heads = {{'Classification','(focal loss)'},[0.67 0.92 0.78],3.5; ...
         {'Calibrated detection','(val threshold \tau)'},[0.96 0.80 0.65],2.55; ...
         {'Open-set reject','(Mahalanobis)'},PURPLE,1.6; ...
         {'Gradient','saliency'},SALMON,0.65};
for k=1:size(heads,1)
    yy = heads{k,3};
    drawbox(9.0,yy-0.32,1.9,0.72,heads{k,1},heads{k,2});
    local_flowarrow(ax,8.6,2.2,9.0,yy+0.04);
end
text(ax,5.5,4.1,'Figure 2.  One backbone, four heads','HorizontalAlignment','center','FontWeight','bold','FontSize',13);
text(ax,7.75,3.0,'shared features','HorizontalAlignment','center','FontAngle','italic','Color',GREY,'FontSize',9);
save_fig(f,'fig2_architecture');

%% ===================== FIGURE 3 — class balance + baselines =============
T  = readtable(fullfile(resDir,'tep_class_distribution.csv'));
Tb = readtable(fullfile(resDir,'baselines_metrics.csv'));
Td = readtable(fullfile(resDir,'deep_baselines_metrics.csv'));
f = figure('Color','w','Position',[100 100 1100 400]);
subplot(1,2,1);
cols = repmat(BLUE,height(T),1); cols(T.faultNumber==0,:) = RED;
b = bar(T.faultNumber, T.n_windows,'FaceColor','flat','EdgeColor','k','LineWidth',0.4);
b.CData = cols;
xlabel('Fault class (0 = normal)'); ylabel('Window count');
title('(a) Tennessee Eastman class balance','FontWeight','bold'); xticks(0:2:20);
subplot(1,2,2);
cnn_f1 = Td.macro_f1(strcmp(Td.model,'CNN1D'));
names = {'PCA-NC','Lin.SVM','RF','MLP','CNN-1D'};
mf1   = [Tb.macro_f1(strcmp(Tb.model,'PCA+NearestCentroid')); ...
         Tb.macro_f1(strcmp(Tb.model,'LinearSVM')); ...
         Tb.macro_f1(strcmp(Tb.model,'RandomForest')); ...
         Tb.macro_f1(strcmp(Tb.model,'MLP')); cnn_f1];
bc = [GREY;GREY;BLUE;BLUE;GREEN];
b2 = bar(mf1,'FaceColor','flat','EdgeColor','k','LineWidth',0.5); b2.CData = bc;
xticklabels(names); ylabel('Macro-F1'); ylim([0 1]);
title('(b) Baseline model comparison','FontWeight','bold');
for i=1:numel(mf1), text(i,mf1(i)+0.02,sprintf('%.2f',mf1(i)),'HorizontalAlignment','center','FontSize',9); end
sgtitle('Figure 3.  Data balance and baseline diagnosis performance','FontWeight','bold','FontSize',13);
save_fig(f,'fig3_data_baselines');

%% ===================== FIGURE 4 — variable saliency =====================
Ti = readtable(fullfile(resDir,'unified_interpretability.csv'));
Ti = sortrows(Ti,'saliency','descend'); Ti = Ti(1:8,:);
lbl = containers.Map( ...
 {'xmeas_21','xmv_10','xmeas_18','xmeas_9','xmeas_19','xmeas_13','xmv_1','xmeas_16'}, ...
 {'Reactor CW outlet temp','Condenser CW valve','Stripper temp','Reactor temp', ...
  'Stripper steam flow','Prod. sep. pressure','D feed valve','Stripper pressure'});
tags = strings(8,1);
for i=1:8
    v = Ti.variable{i};
    if isKey(lbl,v), tags(i)=sprintf('%s  -  %s',v,lbl(v)); else, tags(i)=v; end
end
f = figure('Color','w','Position',[100 100 850 460]);
barh(flipud(Ti.saliency),'FaceColor',BLUE,'EdgeColor','k','LineWidth',0.5);
yticks(1:8); yticklabels(flipud(tags)); set(gca,'FontSize',9);
xlabel('Input-gradient saliency (normalized)');
title('Figure 4.  Variable attribution on Tennessee Eastman','FontWeight','bold','FontSize',13);
save_fig(f,'fig4_saliency');

%% ===================== FIGURE 5 — calibrated operating point ============
Tdet = readtable(fullfile(resDir,'unified_detection.csv'));
calib = [Tdet.FDR(1), Tdet.FAR(1)];        % from CSV: 0.90 , 0.06
delay = Tdet.mean_detection_delay_min(1);  % 47.8 min
naive = [0.96, 0.90];                      % pre-fix naive-argmax reference
f = figure('Color','w','Position',[100 100 760 440]);
Y = [naive; calib]';                       % rows = [FDR;FAR], cols=[naive calib]
b = bar(Y,'grouped','EdgeColor','k','LineWidth',0.5);
b(1).FaceColor = RED; b(2).FaceColor = GREEN;
xticklabels({'Fault-detection rate (FDR)','False-alarm rate (FAR)'});
ylim([0 1.05]); ylabel('Rate');
legend({'Naive argmax','Validation-calibrated (\tau)'},'Box','off','Location','northeast');
title({'Figure 5.  Calibrated threshold fixes the false-alarm problem', ...
       sprintf('(mean detection delay %.1f min)',delay)},'FontWeight','bold','FontSize',12);
xt=[0.855 1.145 1.855 2.145]; yt=[naive(1) calib(1) naive(2) calib(2)];
for i=1:4, text(xt(i),yt(i)+0.015,sprintf('%.2f',yt(i)),'HorizontalAlignment','center','FontSize',9); end
save_fig(f,'fig5_operating_point');

%% ===================== FIGURE 6 — open-set AUROC by score ===============
To = readtable(fullfile(resDir,'unified_openset.csv'));
scores = {'Max softmax','Energy','Entropy','Mahalanobis (features)'};
auroc  = [To.MSP_AUROC(1), To.energy_AUROC(1), To.entropy_AUROC(1), To.mahalanobis_AUROC(1)];
bc = [GREY;GREY;GREY;GREEN];
f = figure('Color','w','Position',[100 100 760 440]);
b = bar(auroc,'FaceColor','flat','EdgeColor','k','LineWidth',0.5); b.CData = bc; hold on;
yline(0.5,'--k','chance','LineWidth',1,'FontSize',8,'LabelHorizontalAlignment','right');
xticklabels(scores); ylabel('AUROC (unknown vs known)'); ylim([0 1]);
for i=1:numel(auroc), text(i,auroc(i)+0.015,sprintf('%.2f',auroc(i)),'HorizontalAlignment','center','FontSize',9); end
title({'Figure 6.  Unknown-fault rejection: feature distance beats softmax', ...
       '(held-out faults 16, 17, 18)'},'FontWeight','bold','FontSize',12);
save_fig(f,'fig6_openset_auroc');

%% ============ OPTIONAL: richer Fig 5/6 if raw arrays exported ===========
% Export arr_fault_score.npy etc. to CSV first, then uncomment.
fs_path = fullfile(resDir,'arr_fault_score.csv');
if isfile(fs_path)
    local_curve_fig5(resDir, outDir, RED, GREEN); %#ok<UNRCH>
end
sc_path = fullfile(resDir,'arr_scores.csv');   % columns: is_unknown, maha, energy, entropy
if isfile(sc_path)
    local_dist_fig6(resDir, outDir, BLUE, RED); %#ok<UNRCH>
end

fprintf('All figures written to: %s\n', outDir);
end

% ======================= helpers =======================
function local_save(f, outDir, name)
    exportgraphics(f, fullfile(outDir,[name '.png']), 'Resolution',300);
    exportgraphics(f, fullfile(outDir,[name '.pdf']), 'ContentType','vector');
end

function local_box(ax,x,y,w,h,t,c)
    rectangle(ax,'Position',[x y w h],'Curvature',0.25,'FaceColor',c,'EdgeColor','k','LineWidth',1.1);
    text(ax,x+w/2,y+h/2,t,'HorizontalAlignment','center','VerticalAlignment','middle','FontWeight','bold','FontSize',9);
end

function local_flowarrow(ax,x1,y1,x2,y2)
    annotation(ax.Parent,'arrow', ...
        dsxy2figxy(ax,[x1 x2]), dsxy2figxy_y(ax,[y1 y2]),'LineWidth',1.3,'Color',[0.2 0.2 0.2]);
end

function out = dsxy2figxy(ax,x)
    p = ax.Position; xl = xlim(ax);
    out = p(1) + (x - xl(1))/(xl(2)-xl(1))*p(3);
end
function out = dsxy2figxy_y(ax,y)
    p = ax.Position; yl = ylim(ax);
    out = p(2) + (y - yl(1))/(yl(2)-yl(1))*p(4);
end

function local_curve_fig5(resDir,outDir,RED,GREEN)
    S = readtable(fullfile(resDir,'arr_fault_score.csv')); % cols: score, is_normal, is_knownfault
    thr = linspace(0,1,200); far=zeros(size(thr)); fdr=zeros(size(thr));
    nrm = S.score(logical(S.is_normal)); flt = S.score(logical(S.is_knownfault));
    for i=1:numel(thr)
        far(i)=mean(nrm>=thr(i)); fdr(i)=mean(flt>=thr(i));
    end
    f=figure('Color','w','Position',[100 100 620 500]);
    plot(far,fdr,'-','Color',GREEN,'LineWidth',2); hold on;
    plot(0.06,0.90,'o','MarkerFaceColor',GREEN,'MarkerEdgeColor','k','MarkerSize',9);
    plot(0.90,0.96,'s','MarkerFaceColor',RED,'MarkerEdgeColor','k','MarkerSize',9);
    xlabel('False-alarm rate'); ylabel('Fault-detection rate');
    legend({'Operating curve','Calibrated (\tau)','Naive argmax'},'Box','off','Location','southeast');
    title('Figure 5.  Detection operating characteristic','FontWeight','bold');
    exportgraphics(f,fullfile(outDir,'fig5_curve.png'),'Resolution',300);
end

function local_dist_fig6(resDir,outDir,BLUE,RED)
    S = readtable(fullfile(resDir,'arr_scores.csv')); % cols: is_unknown, maha
    known = S.maha(S.is_unknown==0); unk = S.maha(S.is_unknown==1);
    f=figure('Color','w','Position',[100 100 620 460]);
    histogram(known,40,'FaceColor',BLUE,'FaceAlpha',0.6,'EdgeColor','none'); hold on;
    histogram(unk,40,'FaceColor',RED,'FaceAlpha',0.6,'EdgeColor','none');
    xlabel('Mahalanobis distance'); ylabel('Count');
    legend({'Known faults','Unknown faults'},'Box','off');
    title('Figure 6.  Open-set score separation','FontWeight','bold');
    exportgraphics(f,fullfile(outDir,'fig6_distribution.png'),'Resolution',300);
end
