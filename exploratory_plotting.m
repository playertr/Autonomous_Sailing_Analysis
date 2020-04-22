% exploratory_plotting.m
% This file is for initial examination of Daniel Flanigan's data from the
% 80-foot Cabron.

load('Aug10'); %load Aug10.mat
% The timestamp is the Utc column. It is represented in fractional days from
% Unix Epoch Time. To get seconds elapsed from the start, we do
% Aug10.time = (Aug10.Utc - Aug10.Utc(2)) * 24 * 60 * 60;

figure(1);
clf
subplot(2,1,1);
hold on
plot(Aug10.time, Aug10.AWA, 'b', 'DisplayName', 'AWA') % Apparent wind angle
plot(Aug10.time, Aug10.TWD, 'r', 'DisplayName', 'TWD') % True wind direction
legend;

subplot(2,1,2);
plot(Aug10.time, Aug10.HDG, 'k', 'DisplayName', 'HDG') % Heading
legend;

%%

% Attempt to subtract wind velocity and boat velocity to get TWA
% Note: heading HDG is way different that course over ground COG. That
% doesn't really make sense, since it is unlikely that there's 60 degrees
% of leeway. Maybe the compass is facing the wrong way?
app_wind_vec = [Aug10.AWS .* cos(Aug10.AWA * pi/180), Aug10.AWS .* sin(Aug10.AWA * pi/180)];
boat_vel_vec = [Aug10.SOG .* cos(Aug10.HDG * pi/180), Aug10.SOG .* sin(Aug10.HDG * pi/180)];

true_wind_vec = app_wind_vec - boat_vel_vec;
TWA = atan2(true_wind_vec(:,2), true_wind_vec(:,1)) * 180/pi;

figure(2);clf;
subplot(2,1,1);
plot(Aug10.time, TWA, 'DisplayName', 'Predicted TWA');
legend;

subplot(2,1,2);
plot(Aug10.time, Aug10.TWA, 'DisplayName', 'Reported TWA');
legend;

% Hmmm, we're off by 50 degrees at times and I'm not sure why.
% Maybe it's heel?
% Note: Botin has a canting keel and maintains relatively low heel.

%%
% Try again, this time using the heel correction for AWA

app_wind_vec = [Aug10.AWS .* cos(Aug10.AWA * pi/180), Aug10.AWS .* sin(Aug10.AWA * pi/180)];
app_wind_vec(:,2) = app_wind_vec(:,2) ./ cos(Aug10.Heel * pi/180); % Correction

true_wind_vec = app_wind_vec;
true_wind_vec(:,1) = true_wind_vec (:,1) - Aug10.BSP;

TWA = atan2(true_wind_vec(:,2) , true_wind_vec(:,1)) * 180/pi;

%error = min( mod(Aug10.TWA - TWA, 360), mod(TWA - Aug10.TWA, 360));
err = TWA - Aug10.TWA;
err = arrayfun(@(x) wrap_180(x), err);
mse = nanmean(err.^2)

lim = [0 25000];
figure(2);clf;
subplot(3,1,1);
plot(Aug10.time, TWA, 'DisplayName', 'Predicted TWA');
xlim(lim);
legend;

subplot(3,1,2);
plot(Aug10.time, Aug10.TWA, 'DisplayName', 'Reported TWA');
xlim(lim);
legend;

subplot(3,1,3);
plot(Aug10.time, err, 'DisplayName', 'Error');
xlim(lim);
ylim([-10 10])
legend;

% Note: the TWA I generated reads about 1 degree too low. I think this
% might be because the wind instruments reporting TWA accounted for mast
% twist. 1 degree of mast twist (pulling the mainsail downwind in the
% back) would cause the vane to under-read by 1 degree, and that amount
% isn't unreasonable.
% Also, much of the error (wrt the reported TWA) is transient, so it may be
% a result of mast swing. It doesn't seem to be correlated with diff(Heel)
% though.

function a = wrap_180(x)
    while x > 180
        x = x-180;
    end
    while x < -180
        x = x+180;
    end
    a = x;
end

