clear;
%rng(1356);
% INITIALIZE DATA
data = csvread('zoo/zoo.data', 0, 1); % data = total samples x total features
data(:, 1:end-1) = mapminmax(data(:, 1:end-1)', 0, 1)'; % normalize data

data = data(randperm(size(data, 1)), :); % shuffle data

data_input = data(:, 1:end-1); % data_input = total training data x total features
data_target = full(ind2vec(data(:, end)'))'; % data_target = total testing data x total features
threshold = 0.5;
iteration = 1;
MAX_ITERATION = 500;

total_samples = size(data_input, 1);
total_input_nodes = size(data_input, 2);
total_output_nodes = size(data_target, 2);

% NETWORK ARCHITECTURE
total_layers = 3; % total_layers: include input and output layer
total_nodes_each_layer = [total_input_nodes, 13, total_output_nodes]; % exclude bias node % total_nodes_each_layer: include input and output layer

learning_rate = 0.3; % range 0-1
momentum = 0.2; % range 0-1

% SPLIT DATA
train_ratio = 0.7;
validation_ratio = 0.15;
test_ratio = 0.15;

% ABSOLUTE SAMPLING
% train_data = data_input(1:round(train_ratio*total_samples), :)';
% validation_data = data_input(1:round(validation_ratio*total_samples), :)';
% test_data = data_input(1:round(test_ratio*total_samples), :)';
% train_target = data_target(1:round(train_ratio*total_samples), :)';
% validation_target = data_target(1:round(validation_ratio*total_samples), :)';
% test_target = data_target(1:round(test_ratio*total_samples), :)';

% STRATIFIED SAMPLING
each_class_count = histcounts(data(:, end));
train_data = [];
validation_data = [];
test_data = [];
train_target = [];
validation_target = [];
test_target = [];
for i = 1: size(each_class_count, 2)
    ith_class_data = [];
    ith_class_data = data(find(data(:, end) == i), :);
    validation_data_count = round(validation_ratio*each_class_count(i));
    test_data_count = round(test_ratio*each_class_count(i));
    train_data_count = each_class_count(i)-validation_data_count-test_data_count;

    fprintf('%d %d %d %d %d\n', each_class_count(i), train_data_count, validation_data_count, test_data_count, (train_data_count+ validation_data_count+ test_data_count));

    target_temp = zeros(each_class_count(i), size(each_class_count, 2));
    target_temp(:, i) = 1;

    train_data = [train_data; ith_class_data(1:train_data_count, 1:end-1)];
    train_target = [train_target; target_temp(1:train_data_count, :)];

    validation_data = [validation_data; ith_class_data((train_data_count+1):(train_data_count+validation_data_count), 1:end-1)];
    validation_target = [validation_target; target_temp(1:validation_data_count, :)];

    test_data = [test_data; ith_class_data((train_data_count+validation_data_count+1):(train_data_count+validation_data_count+test_data_count), 1:end-1)];
    test_target = [test_target; target_temp(1:test_data_count, :)];
end

train_data = train_data';
validation_data = validation_data';
test_data = test_data';
train_target = train_target';
validation_target = validation_target';
test_target = test_target';

% INITIALIZE WEIGHT AND BIAS
min_weight_rand = -1;
max_weight_rand = 1;
min_bias_rand = -1;
max_bias_rand = 1;
weight = cell(1, total_layers);
bias = cell(1, total_layers);
for l = 2:total_layers
    weight{l} = rand(total_nodes_each_layer(l), (total_nodes_each_layer(l-1)))*(max_weight_rand-min_weight_rand)+min_weight_rand;
    bias{l} = rand(total_nodes_each_layer(l), 1)*(max_bias_rand-min_bias_rand)+min_bias_rand;
end

bigDelta_weight_old = cell(1, total_layers);
bigDelta_bias_old = cell(1, total_layers);

% VALIDATION STEP
validation_iteration_limit = 10;
validation_counter = 0;
train_error_each_iteration = zeros(MAX_ITERATION, 1);
validation_error_each_iteration = zeros(MAX_ITERATION, 1);

% ITERATION
while iteration < MAX_ITERATION
    % FORWARD PASS
    activation = cell(1, total_layers);
    activation{1} = train_data;
    for l = 2:total_layers
        activation{l} = (weight{l}*activation{l-1}) + repmat(bias{l}, 1, size(activation{l-1}, 2));
        activation{l} = 1./(1+exp(activation{l}*-1));
    end

    % VALIDATION STEP
    train_error_each_iteration(iteration, 1) = mean(mean((1/2) * (train_target - activation{total_layers}).^2, 1));

    [val_err, accuracy] = test(weight, total_layers, threshold, bias, validation_target, validation_data, 'VALIDATION', 1);
    validation_error_each_iteration(iteration, 1) = val_err;
    
    if iteration > 1
        if  validation_error_each_iteration(iteration, 1) > validation_error_each_iteration(iteration-1, 1)% && train_error_each_iteration(iteration, 1) < train_error_each_iteration(iteration-1, 1)
            validation_counter = validation_counter + 1;
            if validation_counter == validation_iteration_limit
                break
            end
        else
            validation_counter = 0;
        end
    end    
    
    % BACKWARD PASS
    delta = cell(1, total_layers);

    if iteration > 1
        bigDelta_weight_old{l} = bigDelta_weight{l};
        bigDelta_bias_old{l} = bigDelta_bias{l};
    end

    bigDelta_weight = cell(1, total_layers);
    bigDelta_bias = cell(1, total_layers);
    
    for l = total_layers:-1:2
        if l == total_layers
            delta{l} = (activation{l} - train_target) .* ((1 - activation{l}) .* activation{l});
        else
            delta{l} = (weight{l+1}' * delta{l+1}) .* ((1 - activation{l}) .* activation{l});
        end
        
        bigDelta_weight{l} =  learning_rate * (delta{l} * (activation{l-1}')) * (1 - momentum);
        bigDelta_bias{l} = learning_rate * delta{l} * (1 - momentum);
        
        if iteration == 1
            bigDelta_weight_old{l} = bigDelta_weight{l};
            bigDelta_bias_old{l} = bigDelta_bias{l};
        end
    end

    % UPDATE WEIGHT AND BIAS
    for l = 2:total_layers
        if iteration == 1
            weight{l} = weight{l} + (momentum * 0) - bigDelta_weight{l};
            bias{l} = bias{l} + (momentum * 0) - sum(bigDelta_bias{l}, 2);
        else
            weight{l} = weight{l} + (momentum .* bigDelta_weight_old{l}) - bigDelta_weight{l};
            bias{l} = bias{l} + (momentum .* sum(bigDelta_bias_old{l}, 2)) - sum(bigDelta_bias{l}, 2);
        end
    end
    

        
    fprintf('iteration %d/%d | Validation Error = %d | Train Error = %d | Validation Counter = %d\n', iteration, MAX_ITERATION, validation_error_each_iteration(iteration, 1), train_error_each_iteration(iteration, 1), validation_counter);
    iteration = iteration + 1;
end

close
subplot(1, 2, 1);
plot([1:MAX_ITERATION]', train_error_each_iteration(:, 1)');
title('Train Data Error');
xlabel('Iterations');
ylabel('Error');
subplot(1, 2, 2);
plot([1:MAX_ITERATION]', validation_error_each_iteration(:, 1)');
title('Validation Data Error');
xlabel('Iterations');
ylabel('Error');
drawnow

test(weight, total_layers, threshold, bias, test_target, test_data, '      TEST', 0);
test(weight, total_layers, threshold, bias, validation_target, validation_data, 'VALIDATION', 0);
test(weight, total_layers, threshold, bias, train_target, train_data, '  TRAINING', 0);

