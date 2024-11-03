import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
import matplotlib.pyplot as plt
import time
from utilities import Utilities
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import *


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader,tokenizer, eval_iters=500):
    decoderLMmodel.eval()
    losses = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (X, Y) in enumerate(data_loader):
            if i >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)
            logits, _ = decoderLMmodel(X)
            logits = logits.view(-1, tokenizer.vocab_size)
            Y = Y.view(-1)

            loss = criterion(logits, Y)
            losses.append(loss.item())
            total_loss += loss.item()

            # Log perplexity every 100 iterations during evaluation
            if (i + 1) % 100 == 0:
                mean_loss = torch.tensor(losses).mean()
                perplexity = torch.exp(mean_loss).item()
                print(f"Evaluation Iteration {i + 1}, Perplexity: {perplexity:.2f}")

    mean_loss = torch.tensor(losses).mean()
    perplexity = torch.exp(mean_loss).item()

    decoderLMmodel.train()
    return perplexity

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--part', type=str, required=True, help='Model type to train (part1, part2 or part3)')
    block_size = 32

    # Parse the command-line arguments
    args = parser.parse_args()
    if args.part == "part1":
           
        #loading vocabulary
        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
        print("Vocabulary size is", tokenizer.vocab_size)
        
        #loading train and test set
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=False)
        
        #initiliaze encoder and clasifier
        encoder = TransformerEncoder(tokenizer.vocab_size, n_embd, n_head, n_layer).to(device)
        classifier = Classifier(n_embd, n_hidden, n_output).to(device)

        # Define optimizer
        optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        #training the encoder and classifier
        accuracies = []
        encoder.train()
        classifier.train()
        start_time = time.time()
        for epoch in range(epochs_CLS):
            total_correct = 0
            total_samples = 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                # Forward pass
                enc_output, _ = encoder(xb)
                cls_embedding = enc_output.mean(dim=1)  # Mean pooling across sequence dimension
                preds = classifier(cls_embedding)

                # Compute loss
                loss = criterion(preds, yb)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute accuracy
                _, predicted = torch.max(preds.data, 1)
                total_correct += (predicted == yb).sum().item()
                total_samples += yb.size(0)
                

            accuracy = 100 * total_correct / total_samples
            accuracies.append(accuracy)
            print(f"Epoch [{epoch+1}/{epochs_CLS}], Accuracy: {accuracy:.2f}%") 
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nModel trained in {elapsed_time} seconds")
        
        plt.plot(range(1, epochs_CLS + 1), accuracies, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Over Epochs")
        plt.grid(True)
        plt.savefig("accuracy_part1.png")
        encoder.eval()
        classifier.eval()
        
        #evaluate on test set
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for xb, yb in test_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                enc_output, _ = encoder(xb)
                cls_embedding = enc_output.mean(dim=1)
                preds = classifier(cls_embedding)
                _, predicted = torch.max(preds.data, 1)
                total_correct += (predicted == yb).sum().item()
                total_samples += yb.size(0)

        final_accuracy = 100 * total_correct / total_samples
        print(f"Final Test Accuracy: {final_accuracy:.2f}%")
        
        #copmute the attention map
        utilities = Utilities(tokenizer, encoder, 1)
        sentence = "This is a test sentence for attention map visualization."
        utilities.sanity_check(sentence, block_size)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in the TransformerEncoder: {count_parameters(encoder)}")
    if args.part == 'part2':
        
        #loading vocab
        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
        print("Vocabulary size is", tokenizer.vocab_size)
        
        #loading train and test sets
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        inputfile2 = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile2, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        inputfile3 = "speechesdataset/test_LM_obama.txt"
        with open(inputfile3, 'r', encoding='utf-8') as f:
            lmtestText2 = f.read()
        inputfile4 = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile4, 'r', encoding='utf-8') as f:
            lmtestText3 = f.read()
        
        #initilaize train and test set
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
        test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=False)

        test_LM_obama_dataset = LanguageModelingDataset(tokenizer, lmtestText2,  block_size)
        test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=False)

        test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, lmtestText3,  block_size)
        test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=False)
        
        #initialize decoder
        decoder = TransformerDecoder(tokenizer.vocab_size, n_embd, n_head, n_layer).to(device)
        optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        #train the decoder
        start_time = time.time()
        decoder.train()
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            logits, _ = decoder(xb)
            logits = logits.view(-1, tokenizer.vocab_size)
            yb = yb.view(-1)

            # Compute loss
            loss = criterion(logits, yb)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nModel trained in {elapsed_time} seconds")
        decoder.eval()
        test_sets = {
            "test_LM_hbush": test_LM_hbush_loader,
            "test_LM_obama": test_LM_obama_loader,
            "test_LM_wbush": test_LM_wbush_loader
        }
        
        #compute perplexity on test sets
        for name, loader in test_sets.items():
            perplexity = compute_perplexity(decoder, loader, tokenizer)
            print(f"Perplexity on {name}: {perplexity:.2f}")
            
        #compute perplexity on train set
        perplexity = compute_perplexity(decoder, train_LM_loader, tokenizer)
        print(f"Perplexity on training set: {perplexity:.2f}")
        
        #evalute the attention maps
        utilities = Utilities(tokenizer, decoder, 2)
        sentence = "This is a test sentence for attention map visualization."
        utilities.sanity_check(sentence, block_size)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in the TransformerDecoder: {count_parameters(decoder)}")
    if args.part == 'part3.1':
        
        #loading vocab
        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
        print("Vocabulary size is", tokenizer.vocab_size)
        
        #loading train and test set
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=False)
        
        #initialize encoder and classifier
        encoder = TransformerEncoder2(tokenizer.vocab_size, n_embd, n_head, n_layer).to(device)
        classifier = Classifier(n_embd, n_hidden, n_output).to(device)

        # Define optimizer
        optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        #train encoder and classifier
        accuracies = []
        encoder.train()
        classifier.train()
        start_time = time.time()
        for epoch in range(epochs_CLS):
            total_correct = 0
            total_samples = 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                # Forward pass
                enc_output, _ = encoder(xb)
                cls_embedding = enc_output.mean(dim=1)  # Mean pooling across sequence dimension
                preds = classifier(cls_embedding)

                # Compute loss
                loss = criterion(preds, yb)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute accuracy
                _, predicted = torch.max(preds.data, 1)
                total_correct += (predicted == yb).sum().item()
                total_samples += yb.size(0)
                

            accuracy = 100 * total_correct / total_samples
            accuracies.append(accuracy)
            print(f"Epoch [{epoch+1}/{epochs_CLS}], Accuracy: {accuracy:.2f}%") 
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nModel trained in {elapsed_time} seconds")
        
        plt.plot(range(1, epochs_CLS + 1), accuracies, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Over Epochs")
        plt.grid(True)
        plt.savefig("accuracy_part3.png")
        encoder.eval()
        classifier.eval()
        total_correct = 0
        total_samples = 0
        
        #evaluate on test set
        with torch.no_grad():
            for xb, yb in test_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                enc_output, _ = encoder(xb)
                cls_embedding = enc_output.mean(dim=1)
                preds = classifier(cls_embedding)
                _, predicted = torch.max(preds.data, 1)
                total_correct += (predicted == yb).sum().item()
                total_samples += yb.size(0)

        final_accuracy = 100 * total_correct / total_samples
        print(f"Final Test Accuracy: {final_accuracy:.2f}%")
        
        #compute attention maps
        utilities = Utilities(tokenizer, encoder, 3)
        sentence = "This is a test sentence for attention map visualization."
        utilities.sanity_check(sentence, block_size)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in the TransformerEncoder: {count_parameters(encoder)}")    
    if args.part == 'part3.2':
        
        #loading vocab
        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
        print("Vocabulary size is", tokenizer.vocab_size)
        
        #loading train and test set
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        inputfile2 = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile2, 'r', encoding='utf-8') as f:
            lmtestText = f.read()
        inputfile3 = "speechesdataset/test_LM_obama.txt"
        with open(inputfile3, 'r', encoding='utf-8') as f:
            lmtestText2 = f.read()
        inputfile4 = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile4, 'r', encoding='utf-8') as f:
            lmtestText3 = f.read()

        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
        test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=False)

        test_LM_obama_dataset = LanguageModelingDataset(tokenizer, lmtestText2,  block_size)
        test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=False)

        test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, lmtestText3,  block_size)
        test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=False)
        
        #initiliaze docoder
        decoder = TransformerDecoder2(tokenizer.vocab_size, n_embd, n_head, n_layer).to(device)
        optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        #train the decoder
        start_time = time.time()
        decoder.train()
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            logits, _ = decoder(xb)
            logits = logits.view(-1, tokenizer.vocab_size)
            yb = yb.view(-1)

            # Compute loss
            loss = criterion(logits, yb)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nModel trained in {elapsed_time} seconds")
        decoder.eval()
        test_sets = {
            "test_LM_hbush": test_LM_hbush_loader,
            "test_LM_obama": test_LM_obama_loader,
            "test_LM_wbush": test_LM_wbush_loader
        }
        #evaluate the perplexity
        for name, loader in test_sets.items():
            perplexity = compute_perplexity(decoder, loader, tokenizer)
            print(f"Perplexity on {name}: {perplexity:.2f}")
            
        
        perplexity = compute_perplexity(decoder, train_LM_loader, tokenizer)
        print(f"Perplexity on training set: {perplexity:.2f}")
        
        #compute attention maps
        utilities = Utilities(tokenizer, decoder, 4)
        sentence = "This is a test sentence for attention map visualization."
        utilities.sanity_check(sentence, block_size)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in the TransformerDecoder: {count_parameters(decoder)}")
if __name__ == "__main__":
    main()
