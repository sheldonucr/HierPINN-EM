function out = model

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');
model.modelPath('.');
model.label('2D.mph');
model.comments(['untitled\n\n']);
model.modelNode.create('comp1');
model.geom.create('geom1', 2);
model.mesh.create('mesh1', 'geom1');
model.geom('geom1').lengthUnit([native2unicode(hex2dec({'00' 'b5'}), 'unicode') 'm']);

% Import parameters and geometry
run('paramSet')

% Physics
model.physics.create('c', 'CoefficientFormPDE', 'geom1');
% Import physics
run('physics')

% Labels
model.label('2D.mph');


% Plot
run('sol')

out = model;