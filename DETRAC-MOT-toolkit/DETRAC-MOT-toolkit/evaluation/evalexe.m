function evalexe(commandline)

if(verLessThan('matlab', '7.14.0'))
    [status, b] = system(commandline);
else
    [status,b] = system(commandline, '');
end