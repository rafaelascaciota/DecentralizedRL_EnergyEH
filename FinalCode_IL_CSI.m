close all; clear; clc
% (!) ---> Glauber Comments
% (*) ---> Rafa Comments

%% Simulation Parameters
% Define the system parameters
beta = 5;           % Maximum number of packets
N = beta-1;         % Frame size (equal to the number of actions)
L = 21*8;           % Packet size (in bits)
tau_C = 1e-3;       % Charging slot length (in seconds)
tau_D = 1e-3;       % Data slot length (in seconds)
Pu_db = -20;
Pu = 10.^(Pu_db./10);
P0 = 1:8;             % Transmit Power of the HAP (in W)
D = 3;              % (!) The distance between u and PB
H = 70;             % (!) The distance between u and HAP
n = 2.7;            % Path-loss exponent
f = 2.5e9;
c = 3e8;            % speed of light 

% Initialize learning parameters
% (*) the paper does not present the used values for this simulation
gamma = 0.1;                    % discount factor
% Rician fading LOS WET phase
kpb_db = 2;

% Define the number of frames to run
num_runs = 10;
num_frames = 4500;             % Total number of frames
num_devices = 4;               % Number of users

% Non-linear Energy Harvesting
% From: Massive Wireless Energy Transfer with Multiple Power Beacons 
% for very large Internet of Things
c0 = 0.2308;             % EH unitless constants
c1 = 5.365;
w = 10.73;               % energy harvesting saturation level

% (!) Section VII, paragraph before VII-A:
% All results are an average of 10 simulation runs. 
% Each run contains 50000 time frames. At the beginning of each run, 
% we have a warm-up period of 5000 and 30000 frames for IRSA-IL and IRSA-JAL,
% respectively, to train users and HAP. 
% We then report results after convergence.

% (!) eta is drawn from the datasheet of Powercast [8], but I couldn't find
% it. Using Elena's function seems OK by now.


%% Inline functions to help with the simulations

pTransmit = tau_D*Pu*L;
Bmax = beta*pTransmit;     % Battery capacity (in Joules) β * k
    
% Define the state space
S = 0:pTransmit:Bmax;
    
% Define the action space
A = 0:N-1;

% Saturation non-linear EH function
g = @(x) 1e-3.*w.*(1 - exp(-c0*x*1e3))./(1 + exp(-c0.*(x.*1e3 - c1)));
% (*) The efficiency of the P2110 is around 50% info from: https://slideplayer.com/slide/2323972/

num_Wframes_IL = 500;       % Number of Warm-Up frames (IL)
num_Qframes_IL = num_frames - num_Wframes_IL;


%% FULL CSI
fprintf('FULL CSI - M = 4\n') 

% Rician fading LOS WET phase
M = 4;                 % Number of PB antennas

tic
for i4 = 1: length(n)
    for i3 = 1:length(num_devices)
        num_devices(i3)
        for i2 = 1:length(kpb_db)
            kpb = 10.^(kpb_db(i2)./10);
            for i1 = 1:length(P0)
                fprintf('Transmit Power of the HAP (in W): %d \n',P0(i1))    
                for i0 = 1:length(num_runs)
                for run = 1:num_runs(i0)
                    Q_table = zeros(num_devices(i3), length(S), length(A));
                    for i = 1:num_devices(i3)
                        Q_table(i,:,:) = rand(length(S), length(A)); % Initialize Q-table for each device
                    end
                    
                    % Start at random states
                    for i = 1:num_devices(i3)
                        state(i,:) = randsample(S,1); % Initialize Q-table for each device
                    end
                    reward = zeros(1, num_devices(i3));
            
                    % explore/exploit parameter
                    epsilon = 0.5;
            
                    % learning rate for warm-up (very slow)
                    alpha = 0.001;
            
                    % Warm up to give a better initial Q-table
                    for j = 1:num_Wframes_IL
                        epsilon_decay = num_Wframes_IL;    % decay epsilon after 'epsilon_decay' frames
                        % Decay epsilon
                        if mod(j, epsilon_decay) == 0
                            epsilon = epsilon*(0.9);
                        end
            
                        % Loop over devices
                        for i = 1:num_devices(i3)                        
                            % Rician fading Full CSI
                            ang = 2*pi*rand;
                            phi = -(0:(M-1))*pi*sin(ang);
                            h_los = sqrt(kpb/(1+kpb))*exp(1i*pi/4)*exp(1i*phi);   %LOS component
                            h_nlos = sqrt(1/(2*(1+kpb)))*(randn(1,M)+1i*randn(1,M));%NLOS component
                            hf = h_los + h_nlos;
                            
                            % Rayleigh fading
                            hi = sqrt(0.5)*(randn + 1i*randn);

                            channel_gain = ((c)^2)/((4*pi*f)^2*(D^n(i4)));
                            P(i) = channel_gain.*P0(i1).*(norm(hf)^2);

                            % check all possible actions (number of tx packets)
                            sendPkt = A;        % A is the action space = number of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nA = 0:idxes-1;
                            sizenA = length(nA);
                            rW = rand;                                      % get 1 uniform random number
            
                            % Action selection
                            % choose either explore or exploit
                            if (rW >= epsilon)   % exploit
                                if sizenA == 1
                                    action = 0;
                                else
                                    % Select the best action based on Q-values
                                    [~, idx] = max(Q_table(i, S==state(i), 1:sizenA));
                                    action = idx-1;
                                end
                            else        % explore
                                % Select a random action
                                if sizenA == 1
                                    action = 0;
                                else
                                    action = randsample(nA,1);
                                end
                            end
            
                            rho(i) = action;                        % rho - no of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - ((rho(i))*pTransmit*(abs(hi)^2)/H^n(i4)); % (!) State in t, battery level for t+1
                            battery_level(battery_level<0) = nan;
                            B_i(i) = min(battery_level, Bmax);      % The battery level of ui at time t
            
                            % Observe the next state
                            nx = floor(B_i(i)/pTransmit);
                            next_state = nx + 1;
                            % update the state
                            state(i) = S(next_state);
                            
                            % check all possible actions (number of tx packets)
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nextA = 0:idxes-1;
                            snA = length(nextA);
            
                            % Find maximum value
                            max_Q(i) = max(Q_table(i, next_state, 1:snA));
            
                        end
                        decodedPackets = simIRSA(num_devices(i3), beta, rho);
                        reward = decodedPackets';
                
                        for i = 1:num_devices(i3)
                            % Update Q value
                            current_Q = Q_table(i, S==state(i), A==rho(i));
                            new_Q = (1 - alpha)*current_Q + alpha*(reward(i) + gamma*max_Q(i));
                            Q_table(i, S==state(i), A==rho(i)) = new_Q;
                        end
                    end
            
                    % Q-learning reward
                    % using the Q-matrix from the end of the warm up
                    
                    % learning rate for operation
                    alpha = 0.1;
            
                    for j = 1:num_Qframes_IL
                        epsilon_decay = num_Qframes_IL;    % decay epsilon after 'epsilon_decay' frames
                        % Decay epsilon
                        if mod(j, epsilon_decay) == 0
                            epsilon = epsilon*(0.9);
                        end
            
                        % Loop over devices
                        for i = 1:num_devices(i3)
                            % Rician fading Full CSI
                            ang = 2*pi*rand;
                            phi = -(0:(M-1))*pi*sin(ang);
                            h_los = sqrt(kpb/(1+kpb))*exp(1i*pi/4)*exp(1i*phi);   %LOS component
                            h_nlos = sqrt(1/(2*(1+kpb)))*(randn(1,M)+1i*randn(1,M));%NLOS component
                            hf = h_los + h_nlos;
                        
                            % Rayleigh fading
                            hi = sqrt(0.5)*(randn + 1i*randn);

                            channel_gain = ((c)^2)/((4*pi*f)^2*(D^n(i4)));
                            P(i) = channel_gain.*P0(i1).*(norm(hf)^2);
    
                            % check all possible actions (number of tx packets)
                            sendPkt = A;        % A is the action space = number of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nA = 0:idxes-1;
                            sizenA = length(nA);
                            rW = rand;                                      % get 1 uniform random number
            
                            % Action selection
                            % choose either explore or exploit
                            if (rW >= epsilon)   % exploit
                                if sizenA == 1
                                    action = 0;
                                else
                                    % Select the best action based on Q-values
                                    [~, idx] = max(Q_table(i, S==state(i), 1:sizenA));
                                    action = idx-1;
                                end
                            else        % explore
                                % Select a random action
                                if sizenA == 1
                                    action = 0;
                                else
                                    action = randsample(nA,1);
                                end
                            end
            
                            rho(i) = action;                        % rho - no of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - ((rho(i))*pTransmit*(abs(hi)^2)/H^n(i4)); % (!) State in t, battery level for t+1
                            battery_level(battery_level<0) = nan;
                            B_i(i) = min(battery_level, Bmax);      % The battery level of ui at time t
            
                            % Observe the next state
                            nx = floor(B_i(i)/pTransmit);
                            next_state = nx + 1;
                            % update the state
                            state(i) = S(next_state);
                            
                            % check all possible actions (number of tx packets)
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nextA = 0:idxes-1;
                            snA = length(nextA);
            
                            % Find maximum value
                            max_Q(i) = max(Q_table(i, next_state, 1:snA));
                        end
                        decodedPackets = simIRSA(num_devices(i3), beta, rho);
                        reward = decodedPackets';
                
                        for i = 1:num_devices(i3)
                            % Update Q value
                            current_Q = Q_table(i, S==state(i), A==rho(i));
                            new_Q = (1 - alpha)*current_Q + alpha*(reward(i) + gamma*max_Q(i));
                            Q_table(i, S==state(i), A==rho(i)) = new_Q;
                        end
                        % number of action
                        act_send(j,:) = sum(rho);
                        % sucess received packets
                        sucess(j,:) = sum(reward);
                    end
                    sucess_act(:,run) = mean(act_send);
                    sucess_frames(:,run) = mean(sucess);
                end
                    avgpotFRsuss(i0,:) = mean(sucess_frames);
                end
                sces_POT(i1,i2,i3,i4,:) = mean(sucess_act);
                rwrd_POT(i1,i2,i3,i4,:) = mean(sucess_frames);
            end
        end
    end
end

elapsedTimefull = toc;
aFvsPfull = avgpotFRsuss;
sucessIDfull = sucess;
sucess_framesIDfull = sucess_frames;
IDrwrd_POTfull = rwrd_POT;
actionIDfull = act_send;
sucess_actIDfull = sucess_act;
IDsces_POTfull = sces_POT;

%% AVG. CSI
fprintf('AVG. CSI - M = 4\n') 

tic
for i4 = 1: length(n)
    for i3 = 1:length(num_devices)
        num_devices(i3)
        for i2 = 1:length(kpb_db)
            kpb = 10.^(kpb_db(i2)./10);
            for i1 = 1:length(P0)
                fprintf('Transmit Power of the HAP (in W): %d \n',P0(i1))
                for i0 = 1:length(num_runs)
                for run = 1:num_runs(i0)
                    Q_table = zeros(num_devices(i3), length(S), length(A));
                    for i = 1:num_devices(i3)
                        Q_table(i,:,:) = rand(length(S), length(A)); % Initialize Q-table for each device
                    end
                    
                    % Start at random states
                    for i = 1:num_devices(i3)
                        state(i,:) = randsample(S,1); % Initialize Q-table for each device
                    end
                    reward = zeros(1, num_devices(i3));
            
                    % explore/exploit parameter
                    epsilon = 0.5;
            
                    % learning rate for warm-up (very slow)
                    alpha = 0.001;
            
                    % Warm up to give a better initial Q-table
                    for j = 1:num_Wframes_IL
                        epsilon_decay = num_Wframes_IL;    % decay epsilon after 'epsilon_decay' frames
                        % Decay epsilon
                        if mod(j, epsilon_decay) == 0
                            epsilon = epsilon*(0.9);
                        end
            
                        % Loop over devices
                        for i = 1:num_devices(i3)
                            % Rician fading Avg CSI
                            ang = 2*pi*rand;
                            phi = -(0:(M-1))*pi*sin(ang);
                            h_los = sqrt(kpb/(1+kpb))*exp(1i*pi/4)*exp(1i*phi);   %LOS component
                            h_nlos = sqrt(1/(2*(1+kpb)))*(randn(1,M)+1i*randn(1,M));%NLOS component
                            upha = abs(conj(h_los)*h_los.')^2;
                            downha = norm(h_los)^2;
                            ha = upha/downha ;
                            
                            % Rayleigh fading
                            hi = sqrt(0.5)*(randn + 1i*randn);

                            channel_gain = ((c)^2)/((4*pi*f)^2*(D^n(i4)));
                            P(i) = channel_gain.*P0(i1).*ha;
                            
                            % check all possible actions (number of tx packets)
                            sendPkt = A;        % A is the action space = number of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nA = 0:idxes-1;
                            sizenA = length(nA);
                            rW = rand;                                      % get 1 uniform random number
            
                            % Action selection
                            % choose either explore or exploit
                            if (rW >= epsilon)   % exploit
                                if sizenA == 1
                                    action = 0;
                                else
                                    % Select the best action based on Q-values
                                    [~, idx] = max(Q_table(i, S==state(i), 1:sizenA));
                                    action = idx-1;
                                end
                            else        % explore
                                % Select a random action
                                if sizenA == 1
                                    action = 0;
                                else
                                    action = randsample(nA,1);
                                end
                            end
            
                            rho(i) = action;                        % rho - no of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (rho(i))*pTransmit*(abs(hi)^2)/H^n(i4); % (!) State in t, battery level for t+1
                            battery_level(battery_level<0) = nan;
                            B_i(i) = min(battery_level, Bmax);      % The battery level of ui at time t
            
                            % Observe the next state
                            nx = floor(B_i(i)/pTransmit);
                            next_state = nx + 1;
                            % update the state
                            state(i) = S(next_state);
                            
                            % check all possible actions (number of tx packets)
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nextA = 0:idxes-1;
                            snA = length(nextA);
            
                            % Find maximum value
                            max_Q(i) = max(Q_table(i, next_state, 1:snA));
            
                        end
                        decodedPackets = simIRSA(num_devices(i3), beta, rho);
                        reward = decodedPackets';
                
                        for i = 1:num_devices(i3)
                            % Update Q value
                            current_Q = Q_table(i, S==state(i), A==rho(i));
                            new_Q = (1 - alpha)*current_Q + alpha*(reward(i) + gamma*max_Q(i));
                            Q_table(i, S==state(i), A==rho(i)) = new_Q;
                        end
                    end
            
                    % Q-learning reward
                    % using the Q-matrix from the end of the warm up
                    
                    % learning rate for operation
                    alpha = 0.1;
            
                    for j = 1:num_Qframes_IL
                        epsilon_decay = num_Qframes_IL;    % decay epsilon after 'epsilon_decay' frames
                        % Decay epsilon
                        if mod(j, epsilon_decay) == 0
                            epsilon = epsilon*(0.9);
                        end
            
                        % Loop over devices
                        for i = 1:num_devices(i3)
                            % Rician fading Avg CSI
                            ang = 2*pi*rand;
                            phi = -(0:(M-1))*pi*sin(ang);
                            h_los = sqrt(kpb/(1+kpb))*exp(1i*pi/4)*exp(1i*phi);   %LOS component
                            h_nlos = sqrt(1/(2*(1+kpb)))*(randn(1,M)+1i*randn(1,M));%NLOS component
                            upha = abs(conj(h_los)*h_los.')^2;
                            downha = norm(h_los)^2;
                            ha = upha/downha;
                            
                            % Rayleigh fading
                            hi = sqrt(0.5)*(randn + 1i*randn);

                            channel_gain = ((c)^2)/((4*pi*f)^2*(D^n(i4)));
                            P(i) = channel_gain.*P0(i1).*ha;
            
                            % check all possible actions (number of tx packets)
                            sendPkt = A;        % A is the action space = number of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nA = 0:idxes-1;
                            sizenA = length(nA);
                            rW = rand;                                      % get 1 uniform random number
            
                            % Action selection
                            % choose either explore or exploit
                            if rW >= epsilon   % exploit
                                if sizenA == 1
                                    action = 0;
                                else
                                    % Select the best action based on Q-values
                                    [~, idx] = max(Q_table(i, S==state(i), 1:sizenA));
                                    action = idx-1;
                                end
                            else        % explore
                                % Select a random action
                                if sizenA == 1
                                    action = 0;
                                else
                                    action = randsample(nA,1);
                                end
                            end
                            rho(i) = action;                        % rho - no of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (rho(i))*pTransmit*(abs(hi)^2)/H^n(i4); % (!) State in t, battery level for t+1
                            battery_level(battery_level<0) = nan;
                            B_i(i) = min(battery_level, Bmax);      % The battery level of ui at time t
            
                            % Observe the next state
                            nx = floor(B_i(i)/pTransmit);
                            next_state= nx + 1;
                            % update the state
                            state(i) = S(next_state);
            
                            % check all possible actions (number of tx packets)
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            nextA = 0:idxes-1;
                            snA = length(nextA);
            
                            % Find maximum value
                            max_Q(i) = max(Q_table(i, next_state, 1:snA));
                        end
                        decodedPackets = simIRSA(num_devices(i3), beta, rho);
                        reward = decodedPackets';
                
                        for i = 1:num_devices(i3)
                            % Update Q value
                            current_Q = Q_table(i, S==state(i), A==rho(i));
                            new_Q = (1 - alpha)*current_Q + alpha*(reward(i) + gamma*max_Q(i));
                            Q_table(i, S==state(i), A==rho(i)) = new_Q;
                        end
                        % number of action
                        act_send(j,:) = sum(rho);
                        % sucess received packets
                        sucess(j,:) = sum(reward);
                    end
                    sucess_act(:,run) = mean(act_send);
                    sucess_frames(:,run) = mean(sucess);
                end
                    avgpotFRsuss(i0,:) = mean(sucess_frames);
                end
                sces_POT(i1,i2,i3,i4,:) = mean(sucess_act);
                rwrd_POT(i1,i2,i3,i4,:) = mean(sucess_frames);
            end
        end
    end
end


elapsedTimefull = toc;
aFvsPavg = avgpotFRsuss;
sucessIDavg = sucess;
sucess_framesIDavg = sucess_frames;
IDrwrd_POTavg = rwrd_POT;
actionIDavg = act_send;
sucess_actIDavg = sucess_act;
IDsces_POTavg = sces_POT;

%% 
fprintf('FULL CSI - M = 8\n') 
M = 8;
tic
for i4 = 1: length(n)
    for i3 = 1:length(num_devices)
        num_devices(i3)
        for i2 = 1:length(kpb_db)
            kpb = 10.^(kpb_db(i2)./10);
            for i1 = 1:length(P0)
                fprintf('Transmit Power of the HAP (in W): %d \n',P0(i1))    
                for i0 = 1:length(num_runs)
                for run = 1:num_runs(i0)
                    Q_table = zeros(num_devices(i3), length(S), length(A));
                    for i = 1:num_devices(i3)
                        Q_table(i,:,:) = rand(length(S), length(A)); % Initialize Q-table for each device
                    end
                    
                    % Start at random states
                    for i = 1:num_devices(i3)
                        state(i,:) = randsample(S,1); % Initialize Q-table for each device
                    end
                    reward = zeros(1, num_devices(i3));
            
                    % explore/exploit parameter
                    epsilon = 0.5;
            
                    % learning rate for warm-up (very slow)
                    alpha = 0.001;
            
                    % Warm up to give a better initial Q-table
                    for j = 1:num_Wframes_IL
                        epsilon_decay = num_Wframes_IL;    % decay epsilon after 'epsilon_decay' frames
                        % Decay epsilon
                        if mod(j, epsilon_decay) == 0
                            epsilon = epsilon*(0.9);
                        end
            
                        % Loop over devices
                        for i = 1:num_devices(i3)                        
                            % Rician fading Full CSI
                            ang = 2*pi*rand;
                            phi = -(0:(M-1))*pi*sin(ang);
                            h_los = sqrt(kpb/(1+kpb))*exp(1i*pi/4)*exp(1i*phi);   %LOS component
                            h_nlos = sqrt(1/(2*(1+kpb)))*(randn(1,M)+1i*randn(1,M));%NLOS component
                            hf = h_los + h_nlos;
                            
                            % Rayleigh fading
                            hi = sqrt(0.5)*(randn + 1i*randn);

                            channel_gain = ((c)^2)/((4*pi*f)^2*(D^n(i4)));
                            P(i) = channel_gain.*P0(i1).*(norm(hf)^2);

                            % check all possible actions (number of tx packets)
                            sendPkt = A;        % A is the action space = number of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nA = 0:idxes-1;
                            sizenA = length(nA);
                            rW = rand;                                      % get 1 uniform random number
            
                            % Action selection
                            % choose either explore or exploit
                            if (rW >= epsilon)   % exploit
                                if sizenA == 1
                                    action = 0;
                                else
                                    % Select the best action based on Q-values
                                    [~, idx] = max(Q_table(i, S==state(i), 1:sizenA));
                                    action = idx-1;
                                end
                            else        % explore
                                % Select a random action
                                if sizenA == 1
                                    action = 0;
                                else
                                    action = randsample(nA,1);
                                end
                            end
            
                            rho(i) = action;                        % rho - no of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - ((rho(i))*pTransmit*(abs(hi)^2)/H^n(i4)); % (!) State in t, battery level for t+1
                            battery_level(battery_level<0) = nan;
                            B_i(i) = min(battery_level, Bmax);      % The battery level of ui at time t
            
                            % Observe the next state
                            nx = floor(B_i(i)/pTransmit);
                            next_state = nx + 1;
                            % update the state
                            state(i) = S(next_state);
                            
                            % check all possible actions (number of tx packets)
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nextA = 0:idxes-1;
                            snA = length(nextA);
            
                            % Find maximum value
                            max_Q(i) = max(Q_table(i, next_state, 1:snA));
            
                        end
                        decodedPackets = simIRSA(num_devices(i3), beta, rho);
                        reward = decodedPackets';
                
                        for i = 1:num_devices(i3)
                            % Update Q value
                            current_Q = Q_table(i, S==state(i), A==rho(i));
                            new_Q = (1 - alpha)*current_Q + alpha*(reward(i) + gamma*max_Q(i));
                            Q_table(i, S==state(i), A==rho(i)) = new_Q;
                        end
                    end
            
                    % Q-learning reward
                    % using the Q-matrix from the end of the warm up
                    
                    % learning rate for operation
                    alpha = 0.1;
            
                    for j = 1:num_Qframes_IL
                        epsilon_decay = num_Qframes_IL;    % decay epsilon after 'epsilon_decay' frames
                        % Decay epsilon
                        if mod(j, epsilon_decay) == 0
                            epsilon = epsilon*(0.9);
                        end
            
                        % Loop over devices
                        for i = 1:num_devices(i3)
                            % Rician fading Full CSI
                            ang = 2*pi*rand;
                            phi = -(0:(M-1))*pi*sin(ang);
                            h_los = sqrt(kpb/(1+kpb))*exp(1i*pi/4)*exp(1i*phi);   %LOS component
                            h_nlos = sqrt(1/(2*(1+kpb)))*(randn(1,M)+1i*randn(1,M));%NLOS component
                            hf = h_los + h_nlos;
                        
                            % Rayleigh fading
                            hi = sqrt(0.5)*(randn + 1i*randn);

                            channel_gain = ((c)^2)/((4*pi*f)^2*(D^n(i4)));
                            P(i) = channel_gain.*P0(i1).*(norm(hf)^2);
    
                            % check all possible actions (number of tx packets)
                            sendPkt = A;        % A is the action space = number of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nA = 0:idxes-1;
                            sizenA = length(nA);
                            rW = rand;                                      % get 1 uniform random number
            
                            % Action selection
                            % choose either explore or exploit
                            if (rW >= epsilon)   % exploit
                                if sizenA == 1
                                    action = 0;
                                else
                                    % Select the best action based on Q-values
                                    [~, idx] = max(Q_table(i, S==state(i), 1:sizenA));
                                    action = idx-1;
                                end
                            else        % explore
                                % Select a random action
                                if sizenA == 1
                                    action = 0;
                                else
                                    action = randsample(nA,1);
                                end
                            end
            
                            rho(i) = action;                        % rho - no of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - ((rho(i))*pTransmit*(abs(hi)^2)/H^n(i4)); % (!) State in t, battery level for t+1
                            battery_level(battery_level<0) = nan;
                            B_i(i) = min(battery_level, Bmax);      % The battery level of ui at time t
            
                            % Observe the next state
                            nx = floor(B_i(i)/pTransmit);
                            next_state = nx + 1;
                            % update the state
                            state(i) = S(next_state);
                            
                            % check all possible actions (number of tx packets)
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nextA = 0:idxes-1;
                            snA = length(nextA);
            
                            % Find maximum value
                            max_Q(i) = max(Q_table(i, next_state, 1:snA));
                        end
                        decodedPackets = simIRSA(num_devices(i3), beta, rho);
                        reward = decodedPackets';
                
                        for i = 1:num_devices(i3)
                            % Update Q value
                            current_Q = Q_table(i, S==state(i), A==rho(i));
                            new_Q = (1 - alpha)*current_Q + alpha*(reward(i) + gamma*max_Q(i));
                            Q_table(i, S==state(i), A==rho(i)) = new_Q;
                        end
                        % number of action
                        act_send(j,:) = sum(rho);
                        % sucess received packets
                        sucess(j,:) = sum(reward);
                    end
                    sucess_act(:,run) = mean(act_send);
                    sucess_frames(:,run) = mean(sucess);
                end
                    avgpotFRsuss(i0,:) = mean(sucess_frames);
                end
                sces_POT(i1,i2,i3,i4,:) = mean(sucess_act);
                rwrd_POT(i1,i2,i3,i4,:) = mean(sucess_frames);
            end
        end
    end
end

elapsedTimefull8 = toc;
aFvsPfull8 = avgpotFRsuss;
sucessIDfull8 = sucess;
sucess_framesIDfull8 = sucess_frames;
IDrwrd_POTfull8 = rwrd_POT;
actionIDfull8 = act_send;
sucess_actIDfull8 = sucess_act;
IDsces_POTfull8 = sces_POT;

%% AVG. CSI
fprintf('AVG. CSI - M = 8\n') 

tic
for i4 = 1: length(n)
    for i3 = 1:length(num_devices)
        num_devices(i3)
        for i2 = 1:length(kpb_db)
            kpb = 10.^(kpb_db(i2)./10);
            for i1 = 1:length(P0)
                fprintf('Transmit Power of the HAP (in W): %d \n',P0(i1))
                for i0 = 1:length(num_runs)
                for run = 1:num_runs(i0)
                    Q_table = zeros(num_devices(i3), length(S), length(A));
                    for i = 1:num_devices(i3)
                        Q_table(i,:,:) = rand(length(S), length(A)); % Initialize Q-table for each device
                    end
                    
                    % Start at random states
                    for i = 1:num_devices(i3)
                        state(i,:) = randsample(S,1); % Initialize Q-table for each device
                    end
                    reward = zeros(1, num_devices(i3));
            
                    % explore/exploit parameter
                    epsilon = 0.5;
            
                    % learning rate for warm-up (very slow)
                    alpha = 0.001;
            
                    % Warm up to give a better initial Q-table
                    for j = 1:num_Wframes_IL
                        epsilon_decay = num_Wframes_IL;    % decay epsilon after 'epsilon_decay' frames
                        % Decay epsilon
                        if mod(j, epsilon_decay) == 0
                            epsilon = epsilon*(0.9);
                        end
            
                        % Loop over devices
                        for i = 1:num_devices(i3)
                            % Rician fading Avg CSI
                            ang = 2*pi*rand;
                            phi = -(0:(M-1))*pi*sin(ang);
                            h_los = sqrt(kpb/(1+kpb))*exp(1i*pi/4)*exp(1i*phi);   %LOS component
                            h_nlos = sqrt(1/(2*(1+kpb)))*(randn(1,M)+1i*randn(1,M));%NLOS component
                            upha = abs(conj(h_los)*h_los.')^2;
                            downha = norm(h_los)^2;
                            ha = upha/downha ;
                            
                            % Rayleigh fading
                            hi = sqrt(0.5)*(randn + 1i*randn);

                            channel_gain = ((c)^2)/((4*pi*f)^2*(D^n(i4)));
                            P(i) = channel_gain.*P0(i1).*ha;
                            
                            % check all possible actions (number of tx packets)
                            sendPkt = A;        % A is the action space = number of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nA = 0:idxes-1;
                            sizenA = length(nA);
                            rW = rand;                                      % get 1 uniform random number
            
                            % Action selection
                            % choose either explore or exploit
                            if (rW >= epsilon)   % exploit
                                if sizenA == 1
                                    action = 0;
                                else
                                    % Select the best action based on Q-values
                                    [~, idx] = max(Q_table(i, S==state(i), 1:sizenA));
                                    action = idx-1;
                                end
                            else        % explore
                                % Select a random action
                                if sizenA == 1
                                    action = 0;
                                else
                                    action = randsample(nA,1);
                                end
                            end
            
                            rho(i) = action;                        % rho - no of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (rho(i))*pTransmit*(abs(hi)^2)/H^n(i4); % (!) State in t, battery level for t+1
                            battery_level(battery_level<0) = nan;
                            B_i(i) = min(battery_level, Bmax);      % The battery level of ui at time t
            
                            % Observe the next state
                            nx = floor(B_i(i)/pTransmit);
                            next_state = nx + 1;
                            % update the state
                            state(i) = S(next_state);
                            
                            % check all possible actions (number of tx packets)
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nextA = 0:idxes-1;
                            snA = length(nextA);
            
                            % Find maximum value
                            max_Q(i) = max(Q_table(i, next_state, 1:snA));
            
                        end
                        decodedPackets = simIRSA(num_devices(i3), beta, rho);
                        reward = decodedPackets';
                
                        for i = 1:num_devices(i3)
                            % Update Q value
                            current_Q = Q_table(i, S==state(i), A==rho(i));
                            new_Q = (1 - alpha)*current_Q + alpha*(reward(i) + gamma*max_Q(i));
                            Q_table(i, S==state(i), A==rho(i)) = new_Q;
                        end
                    end
            
                    % Q-learning reward
                    % using the Q-matrix from the end of the warm up
                    
                    % learning rate for operation
                    alpha = 0.1;
            
                    for j = 1:num_Qframes_IL
                        epsilon_decay = num_Qframes_IL;    % decay epsilon after 'epsilon_decay' frames
                        % Decay epsilon
                        if mod(j, epsilon_decay) == 0
                            epsilon = epsilon*(0.9);
                        end
            
                        % Loop over devices
                        for i = 1:num_devices(i3)
                            % Rician fading Avg CSI
                            ang = 2*pi*rand;
                            phi = -(0:(M-1))*pi*sin(ang);
                            h_los = sqrt(kpb/(1+kpb))*exp(1i*pi/4)*exp(1i*phi);   %LOS component
                            h_nlos = sqrt(1/(2*(1+kpb)))*(randn(1,M)+1i*randn(1,M));%NLOS component
                            upha = abs(conj(h_los)*h_los.')^2;
                            downha = norm(h_los)^2;
                            ha = upha/downha;
                            
                            % Rayleigh fading
                            hi = sqrt(0.5)*(randn + 1i*randn);

                            channel_gain = ((c)^2)/((4*pi*f)^2*(D^n(i4)));
                            P(i) = channel_gain.*P0(i1).*ha;
            
                            % check all possible actions (number of tx packets)
                            sendPkt = A;        % A is the action space = number of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            % new action space (i.e., limiting the number of replicas, given
                            % the battery level)
                            nA = 0:idxes-1;
                            sizenA = length(nA);
                            rW = rand;                                      % get 1 uniform random number
            
                            % Action selection
                            % choose either explore or exploit
                            if rW >= epsilon   % exploit
                                if sizenA == 1
                                    action = 0;
                                else
                                    % Select the best action based on Q-values
                                    [~, idx] = max(Q_table(i, S==state(i), 1:sizenA));
                                    action = idx-1;
                                end
                            else        % explore
                                % Select a random action
                                if sizenA == 1
                                    action = 0;
                                else
                                    action = randsample(nA,1);
                                end
                            end
                            rho(i) = action;                        % rho - no of replicas
                            battery_level = state(i) + (g(P(i))*tau_C) - (rho(i))*pTransmit*(abs(hi)^2)/H^n(i4); % (!) State in t, battery level for t+1
                            battery_level(battery_level<0) = nan;
                            B_i(i) = min(battery_level, Bmax);      % The battery level of ui at time t
            
                            % Observe the next state
                            nx = floor(B_i(i)/pTransmit);
                            next_state= nx + 1;
                            % update the state
                            state(i) = S(next_state);
            
                            % check all possible actions (number of tx packets)
                            battery_level = state(i) + (g(P(i))*tau_C) - (sendPkt)*pTransmit*(abs(hi)^2)/H^n(i4);
                            battery_level(battery_level<0) = nan;
                            [~, idxes] = min(battery_level);
                            
                            nextA = 0:idxes-1;
                            snA = length(nextA);
            
                            % Find maximum value
                            max_Q(i) = max(Q_table(i, next_state, 1:snA));
                        end
                        decodedPackets = simIRSA(num_devices(i3), beta, rho);
                        reward = decodedPackets';
                
                        for i = 1:num_devices(i3)
                            % Update Q value
                            current_Q = Q_table(i, S==state(i), A==rho(i));
                            new_Q = (1 - alpha)*current_Q + alpha*(reward(i) + gamma*max_Q(i));
                            Q_table(i, S==state(i), A==rho(i)) = new_Q;
                        end
                        % number of action
                        act_send(j,:) = sum(rho);
                        % sucess received packets
                        sucess(j,:) = sum(reward);
                    end
                    sucess_act(:,run) = mean(act_send);
                    sucess_frames(:,run) = mean(sucess);
                end
                    avgpotFRsuss(i0,:) = mean(sucess_frames);
                end
                sces_POT(i1,i2,i3,i4,:) = mean(sucess_act);
                rwrd_POT(i1,i2,i3,i4,:) = mean(sucess_frames);
            end
        end
    end
end

elapsedTimefull8 = toc;
aFvsPavg8 = avgpotFRsuss;
sucessIDavg8 = sucess;
sucess_framesIDavg8 = sucess_frames;
IDrwrd_POTavg8 = rwrd_POT;
actionIDavg8 = act_send;
sucess_actIDavg8 = sucess_act;
IDsces_POTavg8 = sces_POT;

%% SAVING RESULTS
% 
% t = datetime('now');
% t.Format = 'dd.MMM_HH.mm';
% fname = sprintf('SIM%s.mat', t);
% save(fname);

% exit

%% LOAD RESULTS
% load('SIM29.Jun_17.41.mat')


%% PLOT RESULTS

figure
plot(P0,IDrwrd_POTfull,'-s','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
hold on
plot(P0,IDrwrd_POTavg,'-s','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
plot(P0,IDrwrd_POTfull8,'-x','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
plot(P0,IDrwrd_POTavg8,'-x','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
hold off
grid on;
axis([0.5 10 0.5 2.7])
ax = gca;
ax.YAxis.FontSize = 12; %for y-axis 
ay = gca;
ay.XAxis.FontSize = 12; %for y-axis
xlabel('Power Beacon Transmission Power [W]','FontSize',12,'Interpreter','latex');  
ylabel('Average Number of Successful Transmissions per Frame','FontSize',10,'Interpreter','latex');
legend('F-CSI, M = 4','A-CSI, M = 4','F-CSI, M = 8','A-CSI, M = 8','FontSize', 10); 

% figure
% plot(aFvsPfull,'-s','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% hold on
% plot(aFvsPavg,'-s','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% plot(aFvsPfull8,'-x','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% plot(aFvsPavg8,'-x','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% hold off
% grid on;
% %axis([1 10 2.45 2.7])
% ax = gca;
% ax.YAxis.FontSize = 12; %for y-axis 
% ay = gca;
% ay.XAxis.FontSize = 12; %for y-axis
% xlabel('Framesize','FontSize',12,'Interpreter','latex');  
% ylabel('Average Number of Successful Transmissions per Frame','FontSize',10,'Interpreter','latex');
% legend('F-CSI, M = 4','A-CSI, M = 4','F-CSI, M = 8','A-CSI, M = 8','FontSize', 10); 

% figure
% plot(n,reshape(IDrwrd_POTfull,[1,length(n)]),'-s','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% hold on
% plot(n,reshape(IDrwrd_POTavg,[1,length(n)]),'-s','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% plot(n,reshape(IDrwrd_POTfull8,[1,length(n)]),'-x','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% plot(n,reshape(IDrwrd_POTavg8,[1,length(n)]),'-x','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% hold off
% grid on;
% % axis([1 10 0 2.7])
% ax = gca;
% ax.YAxis.FontSize = 12; %for y-axis 
% ay = gca;
% ay.XAxis.FontSize = 12; %for y-axis
% xlabel('Path-Loss Exponent','FontSize',12,'Interpreter','latex');  
% ylabel('Average Number of Successful Transmissions per Frame','FontSize',10,'Interpreter','latex');
% legend('F-CSI, M = 4','A-CSI, M = 4','F-CSI, M = 8','A-CSI, M = 8','FontSize', 10); 

figure
plot(reshape(IDrwrd_POTfull,[1,length(num_devices)]),'-s','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
hold on
plot(reshape(IDrwrd_POTavg,[1,length(num_devices)]),'-s','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
plot(reshape(IDrwrd_POTfull8,[1,length(num_devices)]),'-x','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
plot(reshape(IDrwrd_POTavg8,[1,length(num_devices)]),'-x','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
hold off
grid on;
axis([1 10 0.4 2.6])
ax = gca;
ax.YAxis.FontSize = 12; %for y-axis 
ay = gca;
ay.XAxis.FontSize = 12; %for y-axis
xlabel('Number of Devices','FontSize',12,'Interpreter','latex');  
ylabel('Average Number of Successful Transmissions per Frame','FontSize',10,'Interpreter','latex');
legend('F-CSI, M = 4','A-CSI, M = 4','F-CSI, M = 8','A-CSI, M = 8','FontSize', 10); 

% figure
% plot(kpb_db ,reshape(IDrwrd_POTfull,[1,length(kpb_db)]),'-s','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% hold on
% plot(kpb_db ,reshape(IDrwrd_POTavg,[1,length(kpb_db)]),'-s','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% plot(kpb_db ,reshape(IDrwrd_POTfull8,[1,length(kpb_db)]),'-x','Color',[0 0.4470 0.7410],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% plot(kpb_db ,reshape(IDrwrd_POTavg8,[1,length(kpb_db)]),'-x','Color',[0.8500 0.3250 0.0980],'LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w')
% hold off
% grid on;
% % axis([1 10 0 2.7])
% ax = gca;
% ax.YAxis.FontSize = 12; %for y-axis 
% ay = gca;
% ay.XAxis.FontSize = 12; %for y-axis
% xlabel('Line-of-Sight factor','FontSize',12,'Interpreter','latex');  
% ylabel('Average Number of Successful Transmissions per Frame','FontSize',10,'Interpreter','latex');
% legend('F-CSI, M = 4','A-CSI, M = 4','F-CSI, M = 8','A-CSI, M = 8','FontSize', 10);  

