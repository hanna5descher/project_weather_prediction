%clear;
% load('dataStation_200705_201704');
% dataTable{1} = station200705;
% dataTable{2} = station201704;
% 
% coordHome = [35.923717, -78.946995]; % home coordinate from Google Maps
% nameState = {'NC','VA','TN','GA','SC'}; % NC and neighboring states
% nameVar = {'WBAN','State','Name','Location','Latitude','Longitude'}; % variables of interest
% dataStation = [];
% for iS = 1 : length(nameState)
%     for iT = 1 : length(dataTable)        
%         tempI = ismember(dataTable{iT}.State,nameState{iS});
%         tempData{iT} = dataTable{iT}(tempI,:);
%         if iS == 1 % filter out "closed" station
%             tempData{iT}(tempData{iT}.WBAN == 93781,:) = [];
%         end
%     end
%     [C,IA,IB] = intersect(tempData{1}.WBAN,tempData{2}.WBAN); % find intersect between weather station datasets
%     dataStationPerState{iS} = tempData{2}(IA,nameVar); % extract station data
%     dataStationPerState{iS}.StateID = ones(size(dataStationPerState{iS},1),1)*iS;
%     dataStationPerState{iS}.relLatitude = coordHome(1)-dataStationPerState{iS}.Latitude; % relative latitude from home: neg-north, pos-south
%     dataStationPerState{iS}.relLongitude = coordHome(2)-dataStationPerState{iS}.Longitude; % relative longitude from home: neg-east, pos-west
%     
%     % append all data
%     dataStation = [dataStation; dataStationPerState{iS}];
% end
% 
% % save data
% writetable(dataStation,'dataStation.csv','Delimiter',',','QuoteStrings',true);

% plot data
color(1,:) = [255 0 127]; % pink
color(2,:) = [0 102 204]; % blue
color(3,:) = [64 64 64]; % green
color(4,:) = [204 0 204]; % magenta
color(5,:) = [0 153 0]; % green
color(6,:) = [0 51 102]; % dark blue
color(7,:) = [255 102 102]; % salmon
color(8,:) = [0 153 153]; % aqua
color(9,:) = [255 153 51]; % orange
color(10,:) = [153 51 255]; % purple

figure(); set(gcf,'Color','w');
states = geoshape(shaperead('usastatehi', 'UseGeoCoords', true));
stateName = 'North Carolina';
nc = states(strcmp(states.Name, stateName));
ax = usamap([30 40],[-90 -75]);
setm(ax, 'FFaceColor', [153 204 255]/255)
geoshow(states,'FaceColor', [204 255 153]/255)
geoshow(nc, 'LineWidth', 1.5, 'FaceColor', [229 204 255]/255)
for k = 1 : 10
    i = find(dataStationOfInterest.cluster == k-1);
    plotm(dataStationOfInterest.Latitude(i,:),dataStationOfInterest.Longitude(i,:),'.','Color',color(k,:)/255,'MarkerSize',8);
    plotm(dataStationCentroids.Latitude(k+1,:),dataStationCentroids.Longitude(k+1,:),'x','Color',color(k,:)/255,'LineWidth',1.5,'MarkerSize',10);
end
plotm(dataStationCentroids.Latitude(1),dataStationCentroids.Longitude(1),'r*','MarkerSize',10); % home
