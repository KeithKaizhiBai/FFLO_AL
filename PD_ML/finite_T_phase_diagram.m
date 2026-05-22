% boundary of tFFLO and normal state, black-line square
T=[0,0.02,0.04,0.06,0.08,0.1,0.13,0.15,0.2,0.3,0.4,0.5,0.55,0.56];
JA=[2.12,1.733,1.5,1.32,1.16,0.9,0.78,0.733,0.7,0.667,0.59,0.4,0.178,0];

% boundary of cFFLO and tFFLO with 1st order phase transition, red dotted line
T1st=[0.01, 0.04,0.05];
JA1st=[0.6,0.6,0.6];

% boundary of cFFLO and tFFLO with 2nd order phase transition, green-line dot
T2nd=[0.06,0.08,0.12,0.16,0.2,0.25,0.3,0.35,0.4];
JA2nd=[0.6,0.6,0.62,0.6277,0.63,0.628,0.617,0.598,0.565];

figure;
plot(T,JA,'-o')
hold on
plot(T1st,JA1st,'-o')
hold on
plot(T2nd,JA2nd,'-o')
hold on