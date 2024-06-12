
function decodedPackets = simISRA(n_users, n_slots, n_replicas)% Parameters
numUsers = n_users;               % Number of users
numSlots = n_slots;               % Number of time slots
numReplicas = n_replicas;         % Number of replicas per packet
    
    % Generate random bipartite graph representing slots selected by each user
    bipartiteGraph = zeros(numUsers, numSlots);
    for user = 1:numUsers
        selectedSlots = randperm(numSlots, numReplicas(user));
        bipartiteGraph(user, selectedSlots) = 1;
    end
    
    % Initialize received packets
    receivedPackets = zeros(numUsers, 1);
    
    % Decoding process
    decodedPackets = zeros(numUsers, 1);
    for user = 1:numUsers
         % Find data slot without collision
        validSlots = find(sum(bipartiteGraph,1) == 1);
        validUser = zeros(1,length(validSlots));
        for VS = 1:length(validSlots)
            validUser(VS) = find(bipartiteGraph(:,validSlots(VS))==1);
        end
        
        if isempty(validSlots)
            %disp("Unable to find a valid slot without collision.")
            break;
        end
        
        mUser = min(validUser,[],"all");
        selectedUser = mUser;
        
        % Decode packet in the selected slot
        decodedPackets(selectedUser) = 1;
        
        % Remove replicas from other slots
        for slot = 1:numSlots
            if bipartiteGraph(selectedUser, slot)
                bipartiteGraph(selectedUser, slot) = 0;
            end
        end

        % Update received packets
        receivedPackets(user) = 1;
    end
end
