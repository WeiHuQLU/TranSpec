import torch
from torch.nn import functional as f

def greedy_search(model, inp, start_symbol, use_cnn=False, use_mlp=False, input_channels=1):
    """
    Model testing function using greedy search.
    
    Args:
        model: Trained model.
        inp: Input spectral data.
        start_symbol: Start symbol for decoding.
        use_cnn: Boolean flag indicating if CNN is used.
        use_mlp: Boolean flag indicating if MLP is used.
        input_channels: Number of input channels.

    Returns:
        greedy_de_predict: List of predicted indices.
    """
    # Forward pass through CNN or MLP if applicable
    if use_cnn:
        x = model.c(inp)
    if use_mlp:
        if input_channels == 1:
            x = model.m(inp.reshape([inp.shape[0], 10, 300]))
        elif input_channels == 2:
            x = model.m(inp.reshape([inp.shape[0], 20, 300]))
    
    # Encode the input
    x = model.encoder(x)
    
    # Initialize the decoder input and terminal flag
    de_input_type = torch.zeros((1, 0), dtype=torch.int64)
    terminal = False
    next_symbol = start_symbol
    
    # Greedy search loop
    while not terminal:
        # Concatenate the next symbol to the decoder input
        de_input_type = torch.cat([de_input_type, torch.tensor([[next_symbol]], dtype=torch.int64)], dim=-1)
        
        # Generate the target mask
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(de_input_type.size(-1))
        
        # Pass through embedding and positional encoding
        emb_out = model.embedding(de_input_type)
        tgt = model.pe(emb_out)
        
        # Pass through the decoder
        tgt = model.decoder(tgt=tgt, memory=x, tgt_mask=tgt_mask, tgt_key_padding_mask=None)
        
        # Predict the next symbol
        pre_type = model.pre_type(tgt)
        pre_type = f.softmax(pre_type, dim=-1)
        
        # Get the index of the highest probability
        prob = pre_type.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        
        # Check for termination condition
        if next_symbol == 2 or len(prob) == 100:
            terminal = True
    
    # Convert the output indices to a list and return
    greedy_de_predict = de_input_type.squeeze(0)[1:].tolist()
    return greedy_de_predict

def threshold_value_search(model, inp, start_symbol, dic_length, threshold_value, candidate_limit, dic, use_cnn=False, use_mlp=False, input_channels=1):
    """
    Model testing function using threshold value search.
    
    Args:
        model: Trained model.
        inp: Input spectral data.
        start_symbol: Start symbol for decoding.
        dic_length: Length of the dictionary.
        threshold_value: Probability threshold for candidate selection.
        candidate_limit: Maximum number of candidate SMILES strings.
        dic: Dictionary for index to character conversion.
        use_cnn: Boolean flag indicating if CNN is used.
        use_mlp: Boolean flag indicating if MLP is used.
        input_channels: Number of input channels.

    Returns:
        candidates_smiles: List of candidate SMILES strings.
        candidates_smiles_probability: List of probabilities for the candidate SMILES strings.
    """
    candidates_smiles = []
    candidates_smiles_probability = []
    
    # Forward pass through CNN or MLP if applicable
    if use_cnn:
        x = model.c(inp)
    if use_mlp:
        if input_channels == 1:
            x = model.m(inp.reshape([inp.shape[0], 10, 300]))
        elif input_channels == 2:
            x = model.m(inp.reshape([inp.shape[0], 20, 300]))
    
    # Encode the input
    x = model.encoder(x)
    
    # Initialize the decoder input and branches
    de_input_type = torch.ones((1, 1), dtype=torch.int64)
    branches = [de_input_type]
    branches_probability = [1]
    
    # Iterate over branches
    for idx, de_i in enumerate(branches):
        de_input = de_i
        smiles_probability = branches_probability[idx]
        terminal = False
        next_symbol = torch.zeros((1, 0), dtype=torch.int64)
        
        # Continue decoding until termination condition is met
        if de_input.size(1) <= 99 and de_input[:, -1] != 2:
            while not terminal:
                de_input = torch.concat([de_input, next_symbol], dim=-1)
                
                # Generate the target mask
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(de_input.size(-1))
                
                # Pass through embedding and positional encoding
                emb_out = model.embedding(de_input)
                tgt = model.pe(emb_out)
                
                # Pass through the decoder
                tgt = model.decoder(tgt=tgt, memory=x, tgt_mask=tgt_mask, tgt_key_padding_mask=None)
                
                # Predict the next symbols
                pre_type = model.pre_type(tgt)
                pre_type = f.softmax(pre_type, dim=-1)
                
                # Get the top-k predictions
                values, indices = torch.topk(pre_type[:, -1, :], k=dic_length, dim=-1)
                values = values.squeeze(0)
                indices = indices.squeeze(0)
                
                # Get the index of the highest probability
                prob = indices[0]
                v = values[0]
                next_word = prob.data
                next_symbol = torch.tensor([[next_word]], dtype=torch.int64)
                
                # Check for termination condition
                if next_word == 2 or pre_type.size(1) >= 99:
                    terminal = True
                else:
                    terminal = False
                    smiles_probability *= v.data.tolist()
                
                # Check if there are more branches to consider
                if pre_type.size(1) <= 99:
                    for ix, vale in enumerate(values[1:]):
                        if vale >= threshold_value:
                            ind = indices[1:][ix].unsqueeze(0).unsqueeze(0)
                            branch_probability, branch = torch.topk(pre_type[:, :-1, :], 1, -1)
                            branch = torch.concat([de_input, ind], dim=-1)
                            branches += [branch]
                            
                            branch_probability = branch_probability.squeeze(-1).squeeze(0)
                            a = 1
                            for i in branch_probability:
                                a *= i
                            branches_probability.append((vale * a).tolist())
        else:
            if de_input[:, -1] == 2:
                de_input = de_input[:, :-1]
        
        de_input = de_input.squeeze(0)[1:].tolist()
        pre_smiles = "".join([dic.get(i) for i in de_input])
        
        # Add the candidate to the list if limit is not exceeded
        if len(candidates_smiles) < candidate_limit:
            candidates_smiles.append(pre_smiles)
            candidates_smiles_probability.append(smiles_probability)
        else:
            break
    
    # Sort candidates by probability
    candidates_smiles_probability, candidates_smiles = (list(t) for t in zip(*sorted(zip(candidates_smiles_probability, candidates_smiles), reverse=True)))
    
    return candidates_smiles, candidates_smiles_probability
