Start = 0
End = 200

for i=Start:End
    % Open input file
    load(char(string('data/') + string(int2str(i)) + string('.mat')))
    % Run solver
    eventTime = EM3Phase_FEMCOMSOL(char(string('data/') + string(int2str(i)) + string('.geo')), J, T, tmax, tstep, plots);
    % Save results
    save(char(string('data/') + string(int2str(i)) + string('.mat')),'sVC','J')
    fclose('all');
end
