function [rgb] = stressColor(stress)
% Assigns color for each stress value
% Tensile = Red
% Neutral = Green
% Compressive = Blue

if(stress <= -4.5e8)                            % Very dark blue
    rgb = [0 0 0.2 ];
elseif(stress > -4.5e8 & stress <= -4e8)        % Dark blue
    rgb = [0 0 0.6];
elseif(stress > -4e8 & stress <= -3.5e8)        % Blue
    rgb = [0 0 1];
elseif(stress > -3.5e8 & stress <= -3e8)        % Light blue
    rgb = [0 0.4 0.8];
elseif(stress > -3e8 & stress <= -2.5e8)        % Sky blue
    rgb = [0 0.5 1];
elseif(stress > -2.5e8 & stress <= -2e8)        % Teal blue
    rgb = [0 0.6 0.6];
elseif(stress > -2e8 & stress <= -1.5e8)        % Teal
    rgb = [0 0.8 0.8];
elseif(stress > -1.5e8 & stress <= -1e8)        % Neon teal
    rgb = [0 1 1];
elseif(stress > -1e8 & stress <= -0.5e8)        % Teal green light
    rgb = [0.2 1 0.6];
elseif(stress > -0.5e8 & stress <= 0)           % Green
    rgb = [0 1 0];
elseif(stress > 0 & stress <= 0.5e8)            % Lime green
    rgb = [0.5 1 0];
elseif(stress > 0.5e8 & stress <= 1e8)          % Dark lime green
    rgb = [0.4 0.8 0];
elseif(stress > 1e8 & stress <= 1.5e8)          % Light yellow
    rgb = [1 1 0.4];
elseif(stress > 1.5e8 & stress <= 2e8)          % Yellow
    rgb = [1 1 0];
elseif(stress > 2e8 & stress <= 2.5e8)          % Caution yellow
    rgb = [0.8 0.8 0];
elseif(stress > 2.5e8 & stress <= 3e8)          % Yellow orange
    rgb = [1 0.7 0.4];
elseif(stress > 3e8 & stress <= 3.5e8)          % Neon Orange
    rgb = [1 0.6 0.2];
elseif(stress > 3.5e8 & stress <= 4e8)          % Orange
    rgb = [1 0.5 0];
elseif(stress > 4e8 & stress <= 4.5e8)          % Sunset orange
    rgb = [0.8 0.4 0];
elseif(stress > 4.5e8 & stress <= 5e8)          % Red
    rgb = [1 0 0];
elseif(stress > 5e8 & stress <= 5.5e8)          % Dark red
    rgb = [0.6 0 0];
else                                            % Verk dark red
    rgb = [0.2 0 0];
end

end

