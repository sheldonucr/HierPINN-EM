function [eventTime] = EM3Phase_FEMCOMSOL(fileName, J, T, tmax, tstep, plots)

addpath('src/');
addpath('src/temp/');

% Define %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run('constants')

tstampsC=[0:tstep:tmax];                % Vector of time stamps for COMSOL
nPlots=3;                               % How many times steps do you want to plot (Evenly dist. between 0 to tmax)
MarkerSize=10;                          % Square marker size on plots
ptstamps=[0:size(tstampsC,2)/(nPlots-1):size(tstampsC,2)];  % Timestamps for plotting (Pointer to stress at the given time stamp)
ptstamps=uint64(ptstamps);
ptstamps(1)=1;
nucTime=inf;                            % Set once void is nucleated. If it remains inf then there's no nucleation
incTime=inf;
groTime=inf;

% End define %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Write file for COMSOL commands
CSFile = fopen('src/temp/paramSet.m','w');
CSFile2 = fopen('src/temp/physics.m','w');
CSFile3 = fopen('src/temp/sol.m','w');



% Read GMESH file
geoFile = fopen(fileName, 'r');                   % Open GMSH file
fgetl(geoFile) ;                                  % Skip 1st line.
buffer = fread(geoFile, [1,Inf], '*char') ;       % Load file into buffer as char
fclose(geoFile) ;                                 % Close file


% Reformat. Removes all GMSH commands and leaves behind numeric data
buffer = regexprep( buffer, 'SetFactory("OpenCASCADE"', '');
buffer = regexprep( buffer, ');\n', '');
buffer = regexprep( buffer, '//', '');
buffer = regexprep( buffer, '+', '');
buffer = regexprep( buffer, 'Rectangle(', '');
buffer = regexprep( buffer, ') = {', ' ');
buffer = regexprep( buffer, '};\n', '');
buffer = regexprep( buffer, ',', '');
buffer = regexprep( buffer, '};', '');
buffer = regexprep( buffer, ');', '');

% - Convert to numeric type.
data = textscan(buffer, '%f %f %f %f %f %f %f');

data(:,1)=[];   % Delete pointer to the segment number (Preserved by row number)
pos(:,1:3)=[data{1}(:),data{2}(:),data{3}(:)];  % Position x,y,z
dim(:,1:3)=[data{4}(:),data{5}(:),data{6}(:)];  % Dimension l,w,h

nSeg=size(pos,1);   % Number of segments



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMSOL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate COMSOL commands

% Set constants
fprintf(CSFile,'model.param.set(''E'', ''%e'');\n',E);
fprintf(CSFile,'model.param.set(''Cu_resis'', ''%e'');\n',Cu_resis);
fprintf(CSFile,'model.param.set(''Z'', ''%f'');\n',Z);
fprintf(CSFile,'model.param.set(''Omega'', ''%e'');\n',Omega);
fprintf(CSFile,'model.param.set(''kB'', ''%e'');\n',kB);
fprintf(CSFile,'model.param.set(''EffB'', ''%e'');\n',EffB);
fprintf(CSFile,'model.param.set(''D0'', ''%e'');\n',D0);
fprintf(CSFile,'model.param.set(''Ea'', ''%e'');\n',Ea);
fprintf(CSFile,'model.param.set(''T'', ''%d[K]'');\n',T);
fprintf(CSFile,'model.param.set(''tmax'', ''%e[s]'');\n',tmax);
fprintf(CSFile,'model.param.set(''tstep'', ''%e[s]'');\n',tstep);
fprintf(CSFile,'model.param.set(''Da'', ''%e'');\n',Da);
fprintf(CSFile,'model.param.set(''kappa'', ''%e'');\n',kappa);

% % Set width of wire. Assuming all wires are same width
% fprintf(CSFile,'model.param.set(''W'', ''%f[um]'');\n',width);

modified=zeros(nSeg,2); % If the segments position and dimensions are modified
                        % Since COMSOL does not accept -ive values for W
                        % and L
                        % i.e. modified(1,1) = W and L modified
% Geometry parameters and current densities
for i=1:nSeg
    % Since COMSOL cannot take negative values for W and L.
    % Change W and L to positive by changing the initial vertex, X and Y
    if(dim(i,1)<0)
        L(i)=abs(dim(i,1));
        X(i)=pos(i,1)+dim(i,1);
        modified(i,1)=1;
    else
        L(i)=dim(i,1);
        X(i)=pos(i,1);
    end
    
    if(dim(i,2)<0)
        W(i)=abs(dim(i,2));
        Y(i)=pos(i,2)+dim(i,2);
        modified(i,2)=1;
    else
        W(i)=dim(i,2);
        Y(i)=pos(i,2);
    end
    % Geometry parameters
    fprintf(CSFile,'model.param.set(''L%d'', ''%f[um]'');\n',i,L(i));
    fprintf(CSFile,'model.param.set(''W%d'', ''%f[um]'');\n',i,W(i));
    fprintf(CSFile,'model.param.set(''X%d'', ''%f[um]'');\n',i,X(i));
    fprintf(CSFile,'model.param.set(''Y%d'', ''%f[um]'');\n',i,Y(i));
    % Current density
    fprintf(CSFile,'model.param.set(''j%d'', ''%e[A/m^2]'');\n',i,J(i));
end

% Draw and label
for i=1:nSeg
    % Draw segments
    fprintf(CSFile,'model.geom(''geom1'').create(''r%d'', ''Rectangle'');\n',i);
    fprintf(CSFile,'model.geom(''geom1'').feature(''r%d'').set(''size'', {''L%d'' ''W%d''});\n',i,i,i);
    fprintf(CSFile,'model.geom(''geom1'').feature(''r%d'').set(''pos'', {''X%d'' ''Y%d''});\n',i,i,i);
    fprintf(CSFile,'model.geom(''geom1'').runPre(''fin'');\n');
    fprintf(CSFile,'model.geom(''geom1'').run(''r%d'');\n',i);
    % Label segments
    fprintf(CSFile,'model.geom(''geom1'').create(''seg%d'', ''ExplicitSelection'');\n',i);
    fprintf(CSFile,'model.geom(''geom1'').feature(''seg%d'').selection(''selection'').set(''r%d'', [1]);\n',i,i);
    fprintf(CSFile,'model.geom(''geom1'').feature(''seg%d'').label(''seg %d'');\n',i,i);
    fprintf(CSFile,'model.geom(''geom1'').run(''seg%d'');\n',i);
end


% Record the four vertices of each segment
vertex=cell(nSeg,4);
boundLine=cell(nSeg,4);
for i=1:nSeg
    vertex{i,1}=[X(i), Y(i)];
    vertex{i,2}=[X(i)+L(i), Y(i)];
    vertex{i,3}=[X(i)+L(i), Y(i)+W(i)];
    vertex{i,4}=[X(i), Y(i)+W(i)];
end


% Check orientation of each segment
for i=1:nSeg
   % Check if the segment is placed vertically or horizontally
   p1=vertex{i,1}; p2=vertex{i,2};
   dist2=sqrt((p2(1)-p1(1))^2 + (p2(2)-p1(2))^2);
   p1=vertex{i,1}; p2=vertex{i,4};
   dist4=sqrt((p2(1)-p1(1))^2 + (p2(2)-p1(2))^2);
   % vertical(i) = 1 if seg i is placed vertically
   if(dist2 < dist4)
       vertical(i)=1;
       width(i)=abs(L(i));
       length(i)=abs(W(i));
   else
       vertical(i)=0;
       width(i)=abs(W(i));
       length(i)=abs(L(i));
   end
end


% Set physics commands
for i=1:nSeg
    if(i ~= 1)
        fprintf(CSFile2,'model.physics(''c'').create(''cfeq%d'', ''CoefficientFormPDE'', 2);\n',i);
        fprintf(CSFile2,'model.physics(''c'').feature(''cfeq%d'').selection.named(''geom1_seg%d'');\n',i,i);
    end
    fprintf(CSFile2,'model.physics(''c'').feature(''cfeq%d'').set(''c'', {''kappa'' ''0'' ''0'' ''kappa''})\n',i);
    fprintf(CSFile2,'model.physics(''c'').feature(''cfeq%d'').set(''f'', ''0'')\n',i);
    if(vertical(i)==1)
        fprintf(CSFile2,'model.physics(''c'').feature(''cfeq%d'').set(''ga'', {''0'' ''-kappa*E*Z*Cu_resis*j%d/Omega''})\n',i,i);
    else
        fprintf(CSFile2,'model.physics(''c'').feature(''cfeq%d'').set(''ga'', {''-kappa*E*Z*Cu_resis*j%d/Omega'' ''0''})\n',i,i);
    end
end



% Plot results (Each segment plotted separately)
for i=1:nSeg
    fprintf(CSFile3,'model.study.create(''std%d'');\n',i);
    fprintf(CSFile3,'model.study(''std%d'').create(''time'', ''Transient'');\n',i);
    fprintf(CSFile3,'model.sol.create(''sol%d'');\n',i);
    fprintf(CSFile3,'model.sol(''sol%d'').study(''std%d'');\n',i,i);
    fprintf(CSFile3,'model.sol(''sol%d'').attach(''std%d'');\n',i,i);
    fprintf(CSFile3,'model.sol(''sol%d'').create(''st%d'', ''StudyStep'');\n',i,i);
    fprintf(CSFile3,'model.sol(''sol%d'').create(''v%d'', ''Variables'');\n',i,i);
    fprintf(CSFile3,'model.sol(''sol%d'').create(''t%d'', ''Time'');\n',i,i);
    fprintf(CSFile3,'model.sol(''sol%d'').feature(''t%d'').create(''fc%d'', ''FullyCoupled'');\n',i,i,i);
    fprintf(CSFile3,'model.sol(''sol%d'').feature(''t%d'').feature.remove(''fcDef'');\n',i,i);
    fprintf(CSFile3,'model.result.create(''pg%d'', ''PlotGroup2D''); \n',i);
    fprintf(CSFile3,'model.result.create(''pg%d_1'', ''PlotGroup1D'');\n',i);
    fprintf(CSFile3,'model.result(''pg%d'').create(''surf%d'', ''Surface''); \n',i,i);
    fprintf(CSFile3,'model.result(''pg%d_1'').create(''lngr%d'', ''LineGraph''); \n',i,i);
    fprintf(CSFile3,'model.result.export.create(''plot%d'', ''Plot'');\n',i);
    fprintf(CSFile3,'model.study(''std%d'').feature(''time'').set(''tlist'', ''range(0,tstep,tmax)'');\n',i);
    fprintf(CSFile3,'model.sol(''sol%d'').attach(''std%d'');\n',i,i);
    fprintf(CSFile3,'model.sol(''sol%d'').feature(''t%d'').set(''tlist'', ''range(0,tstep,tmax)'');\n',i,i);
    fprintf(CSFile3,'model.sol(''sol%d'').runAll;\n',i);
    
    fprintf(CSFile3,'model.result.dataset.create(''cln%d'', ''CutLine2D'');\n',i);
    if(vertical(i)==0)
        cLine(i,1)=vertex{i,1}(1,1);
        cLine(i,2)=vertex{i,1}(1,2)+(vertex{i,4}(1,2)-vertex{i,1}(1,2))/2;
        cLine(i,3)=vertex{i,2}(1,1);
        cLine(i,4)=cLine(i,2);
    else
        cLine(i,1)=vertex{i,1}(1,1)+(vertex{i,2}(1,1)-vertex{i,1}(1,1))/2;
        cLine(i,2)=vertex{i,1}(1,2);
        cLine(i,3)=cLine(i,1);
        cLine(i,4)=vertex{i,4}(1,2);
    end
    fprintf(CSFile3,'model.result.dataset(''cln%d'').set(''genpoints'', {''%f'' ''%f''; ''%f'' ''%f''});\n',i,cLine(i,1),cLine(i,2),cLine(i,3),cLine(i,4));
    fprintf(CSFile3,'model.result(''pg%d'').set(''looplevel'', {''1''});\n',i);
    fprintf(CSFile3,'model.result(''pg%d'').feature(''surf%d'').set(''resolution'', ''normal'');\n',i,i);
    fprintf(CSFile3,'model.result(''pg%d_1'').set(''data'', ''cln%d'');\n',i,i);
    fprintf(CSFile3,'model.result(''pg%d_1'').set(''ylabel'', ''Dependent variable u (1)'');\n',i);
    fprintf(CSFile3,'model.result(''pg%d_1'').set(''xlabel'', ''Arc length'');\n',i);
    fprintf(CSFile3,'model.result(''pg%d_1'').set(''ylabelactive'', false);\n',i);
    fprintf(CSFile3,'model.result(''pg%d_1'').set(''xlabelactive'', false); \n',i);
    fprintf(CSFile3,'model.result(''pg%d_1'').feature(''lngr%d'').set(''resolution'', ''normal'');\n',i,i);
    fprintf(CSFile3,'model.result.export(''plot%d'').set(''plotgroup'', ''pg%d_1'');\n',i,i);
    fprintf(CSFile3,'model.result.export(''plot%d'').set(''filename'', ''./temp/plot%d.txt'');\n',i,i);
    fprintf(CSFile3,'model.result.export(''plot%d'').set(''plot'', ''lngr%d'');\n',i,i);
    fprintf(CSFile3,'model.result.export(''plot%d'').run;\n',i);    
end

fclose(CSFile);  % Close paramSet.m file
fclose(CSFile2); % Close physics.m file
fclose(CSFile3); % Close plot.m file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END COMSOL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% RUN SOLVERS!(NUCLEATION) %%%%%%%%%%%%%%%%%%%%%%%%%%%


% Run COMSOL
run('src/COMSOL_Nucleation2D()');

%%%%%%%%%%%%%%%%%%%%%%%%%% END RUN SOLVERS! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% PARSE COMSOL RESULTS (CHECK FOR VOID NUCLEATION) %%%%%%%%%%%%%%%%%
% If void is detected then we can run growth phase

% Extract results from plot.txt files
nodesC=cell(nSeg,1); sVC=cell(nSeg,1);                 % Initialize
for i=1:nSeg
    fileName=['src/temp/plot', num2str(i), '.txt'];       % Set filename
    plotFile = fopen(fileName, 'r');                   % Open plot file
    for skip=1:8                                       % Skip first 8 lines
        fgetl(plotFile) ;                              
    end
    buffer = fread(plotFile, [1,Inf], '*char') ;       % Load file into buffer as char
    fclose(plotFile) ;                                 % Close file
    data = textscan(buffer, '%f %f');                  % Convert to numeric type and store in data
    data=[data{1,1}(:), data{1,2}(:)];                 % Convert from cell to matrix
    
    % Extract x and y coordinates of all nodes in seg i
    % nodesC{segment=1,1}{rows=nodes,columns=x,y coordinates)
    
    j=1;
    while(j ~= 0)
        if(vertical(i)==0)          % Horizontal segments
            nodesC{i,1}(j,1)=cLine(i,1)+data(j,1);
            nodesC{i,1}(j,2)=cLine(i,2);
        else                        % Vertical segments
            nodesC{i,1}(j,1)=cLine(i,1);
            nodesC{i,1}(j,2)=cLine(i,2)+data(j,1);
        end
        nNodesC(i)=j;               % Number of nodes in seg i (COMSOL)
        if(data(j,1)<data(j+1,1))   % Increment j
            j=j+1;
        else
            j=0;
        end
    end
    
    xxx = linspace(data(1,1),data(nNodesC(i),1),nNodesC(i));
    for k=1:size(tstampsC,2)
        data((k-1)*nNodesC(i)+1:(k-1)*nNodesC(i)+nNodesC(i),2)= interp1(data((k-1)*nNodesC(i)+1:(k-1)*nNodesC(i)+nNodesC(i),1),data((k-1)*nNodesC(i)+1:(k-1)*nNodesC(i)+nNodesC(i),2),xxx);
    end
    
    % Stress vector sVC{segment=i,1}(rows = nodes, columns=tsteps)
    sVC{i,1}=reshape(data(:,2)',nNodesC(i),size(tstampsC,2)); 
end


% Detect void (node >= critStress) and calculate tstamp(critStress)
voidSegNum=0;                       % Segments with void
voidNodeNum=0;                      % The node where the void is formed
nVoids=0;                           % Number of segments with voids
for t=1:size(tstampsC,2)            % Time stamp tstampsC(t)
    if(voidSegNum ~= 0)
        break
    end
    for i=1:nSeg                    % Segment i
        if(sVC{i,1}(1,t) >= critStress)
                % Void nucleated
                fprintf('Void nucleated in timestep %d out of %d timesteps \n', t, size(tstampsC,2));
                nVoids=nVoids+1;                                    % Increment the number of voids found
                nucTime = tstampsC(t);                              % Record nucleation time
                voidLocation(nVoids,1:2) = nodesC{i,1}(1,:);             % Record coordinates of the void
                voidSegNum(nVoids) = i;                             % Record the segment # of the void
                voidNodeNum(nVoids) = 1;                            % Record the node # of the void
        elseif(sVC{i,1}(nNodesC(i),t) >= critStress)
                % Void nucleated
                fprintf('Void nucleated in timestep %d out of %d timesteps \n', t, size(tstampsC,2));
                nVoids=nVoids+1;                                    % Increment the number of voids found
                nucTime = tstampsC(t);                              % Record nucleation time
                voidLocation(nVoids,1:2) = nodesC{i,1}(nNodesC(i),:);    % Record coordinates of the void
                assignin('base','voidLocation',voidLocation);       % Save location to workspace
                voidSegNum(nVoids) = i;                             % Record the segment # of the void
                voidNodeNum(nVoids) = nNodesC(i);                   % Record the node # of the void
        end
    end
end
if(nVoids ~= 0)
    assignin('base','voidLocation',voidLocation);       % Save location to workspace
end
assignin('base','nVoids',nVoids);       % Save location to workspace

if(plots)
    % Plot nucleation results
    for p=1:nPlots
        figure()
        for i=1:nSeg
            for j=1:nNodesC(i)
                stress = sVC{i,1}(j,ptstamps(p));
                sColor = stressColor(stress);
                plot(nodesC{i,1}(j,1),nodesC{i,1}(j,2),'s','MarkerSize',MarkerSize,'MarkerFaceColor',sColor,'MarkerEdgeColor',sColor)
                hold on
            end
        end
        if(tstampsC(ptstamps(p)) >= nucTime)
            plot(voidLocation(:,1),voidLocation(:,2),'or','MarkerSize',40)
            %legend('VOID NUCLEATED')
            str = 'VOID NUCLEATED';
            annotation('textbox',[.2 .5 .3 .3], 'String', str, 'FitBoxToText', 'on')
            hold on;
        end
        axis equal
    end
end


%assignin('base','nodesC',nodesC);
assignin('base','sVC',sVC);

%%%%%%%%%%%%%%%%%%%%%% END PARSE COMSOL RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% RUN INCUBATION & GROWTH PHASE %%%%%%%%%%%%%%%%%%%%%%%%

% % Check if stress saturates
% if(plots)
%     figure()
%     if(voidSegNum == 0)                                     % If no void is found, then show the 1st node of 1st segment
%         plot(tstampsC(1,:), sVC{1,1}(1,:));
%         %plot(tstampsC(1,:), sVC{2,1}(1,:));
%         hold on
%         plot([0,tmax], [critStress,critStress], 'r');
%         legend('1st Node', 'Critical Stress')
%     else                                                    % Else show the node where the void is located
%         plot(tstampsC(1,:), sVC{voidSegNum(1),1}(voidNodeNum(1),:));
%         hold on
%         plot([0,tmax], [critStress,critStress], 'r');
%         legend('Nucleated Node', 'Critical Stress')
%     end
%     title('STRESS SATURATION')
% end


% Plot transient stress of first and last node of each segment
if(plots)
    figure()
    legCount = 1;
    for(i=1:nSeg)
        plot(tstampsC(1,:), sVC{i,1}(1,:));
        hold on
        leg(legCount,:) = sprintf('Segment %d First Node', i);
        legCount = legCount + 1;
        plot(tstampsC(1,:), sVC{i,1}(nNodesC(i),:));
        leg(legCount,:) = sprintf('Segment %d Last Node ', i);
        legCount = legCount + 1;
        hold on
    end
    plot([0,tmax], [critStress,critStress], 'r');
    leg(legCount,:) = sprintf('CRITICAL STRESS     ');
    legend(leg);
    title('Transient Stress')
end


% If void is nucleated, then calculate incubation and growth time
if(nucTime ~= inf)
    % Convert wire dimensions to meters, since constants are in meters
    lengthMeters = length/(1e6);
    widthMeters = width/(1e6);
    heightMeters = height/(1e6);
    
    % The narrowest segment at the void location will be selected as main branch (critical seg)
    minWidth = 100000000;
    for i=1:nVoids
        if(width(voidSegNum(i)) < minWidth)
            MB = voidSegNum(i);
            minWidth = width(voidSegNum(i));
        end
    end

    % Calculate resistance of main branch
    MBresis=Cu_resis*lengthMeters(MB)/(widthMeters(MB)*heightMeters);
    deltaResis=failResis*MBresis;
    
    % Effective total J-velocity w.r.t. main branch
    Jvel = 0;                           % Initialize J-velocity
    for i=1:nVoids
        if(voidNodeNum(i) ~= 1)         % Check if J affects void growth positively or negatively
            effect = -1;
        else
            effect = 1;
        end
        Jvel = Jvel+(effect*J(voidSegNum(i))*widthMeters(voidSegNum(i))/widthMeters(MB));
    end
    
    vel=Da*E*Z*Cu_resis*abs(Jvel)/(kB*T);  
    
    
    % Calculate incubation time
    incTime=(viaDiam/(1e6))/vel;
    %assignin('base','incTime',incTime);

    % Calculate growth time
    groTime=deltaResis/(vel*((Ta_resis/(hTa*(2*heightMeters+widthMeters(MB))))-Cu_resis/(heightMeters*widthMeters(MB))));
    %assignin('base','groTime',groTime);
end


%%%%%%%%%%%%%%%%%%%% END INCUBATION & GROWTH PHASE %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save solutions
eventTime=[nucTime, incTime, groTime];

pause(3)
fclose('all');
pause(3)
delete src/temp/plot*.txt  % Delete all plot files. Results are already loaded into variables.
delete src/temp/paramSet.m
delete src/temp/physics.m
delete src/temp/sol.m

end % End function