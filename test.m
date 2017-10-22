function [err, accuracy] = test(weight, total_layers, threshold, bias, data_target, data_input, text, is_validation_step)
    activation = cell(1, total_layers);
    activation{1} = data_input;
    for l = 2:total_layers
        activation{l} = (weight{l}*activation{l-1}) + repmat(bias{l}, 1, size(activation{l-1}, 2));
        activation{l} = 1./(1+exp(activation{l}*-1)); % activation function: sigmoid
    end
    
    use_threshold = 0;
    if use_threshold == 1
        predicted = (activation{total_layers}>threshold);
    else
        predicted = zeros(size(activation{total_layers}, 1), size(activation{total_layers}, 2));
        for i=1:size(activation{total_layers}, 2)
            predicted(:, i) = (activation{total_layers}(:, i) == max(activation{total_layers}(:, i)));
        end
    end
    false_count = 0;
    true_count = 0;
    for i = 1:size(data_input, 2)
        if predicted(:, i) == data_target(:, i)
            %fprintf('Sample %d TRUE\n', i);
            true_count = true_count + 1;
        else
            %fprintf('Sample %d FALSE | Actual = %d | Predicted = %d\n', i, find(data_target(:, i)==1), find(activation{total_layers}(:, i)==max(activation{total_layers}(:, i))));
            false_count = false_count + 1;
        end
    end
    err = mean(mean((1/2) * (data_target - activation{total_layers}).^2, 1));
    accuracy = round(true_count/size(data_input, 2)*100);
    %accuracy = (size(data_input, 2)-false_count)/size(data_input, 2);
    if is_validation_step == 0
        fprintf('%s Accuracy = %d/%d = %d percent | Error = %d\n', text, true_count, size(data_input, 2), accuracy, err);
    end
end