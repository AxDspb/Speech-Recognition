import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import matplotlib.pyplot as plt
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Transformer, TransformerDecoder
from utilities import Utilities

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
n_head = 2
n_layer = 4


eval_interval = 100
max_iters = 500
eval_iters = 200

n_input = 64
n_hidden = 100
n_output = 3
epochs_CLS = 15

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. 
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
    data, labels = zip(*batch)
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
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
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ 
    Compute the perplexity of the decoderLMmodel on the data in data_loader.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y)
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()

    decoderLMmodel.train()
    return perplexity

def main():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

  
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)


    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    args = parser.parse_args()
    
    #Part 1
    if args.model == "part1":
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

        model = Transformer(tokenizer.vocab_size, n_embd, n_layer, n_hidden, n_output, block_size, n_head).to(device)

        m = model.to(device)
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        trainAccuracies = []
        testAccuracies = []
        # epochs_CLS
        for epoch in range(15):
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                predictions, attentionMap = model(xb)
                loss = criterion(predictions, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs_CLS}], Loss: {loss.item():.4f}")

            trainAccuracy = compute_classifier_accuracy(model, train_CLS_loader)
            trainAccuracies.append(trainAccuracy)
            testAccuracy = compute_classifier_accuracy(model, test_CLS_loader)
            testAccuracies.append(testAccuracy)
            print(f"Classification Accuracy after epoch {epoch + 1}: {trainAccuracy:.2f}%")

        plt.figure()
        plt.plot(range(1, epochs_CLS + 1), trainAccuracies, color = 'blue', label= "Training Accuracy")
        plt.plot(range(1, epochs_CLS + 1), testAccuracies, color = 'red', label= "Test Accuracy")
        plt.title("Classification Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig("CLSGraphs.png")

        test_accuracy = compute_classifier_accuracy(model, test_CLS_loader)
        print(f"Final Test Accuracy for Classification Task: {test_accuracy:.2f}%")

        #Sanity Check
        sentence = "Last week was Holloween."
        utilities = Utilities(tokenizer, model)
        utilities.sanity_check(sentence, block_size)



    #Part 2
    if args.model == "part2":
                
        decoder_model = TransformerDecoder(tokenizer.vocab_size, n_embd, n_layer, block_size, n_hidden, n_head).to(device)
        
        optimizer = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        trainPerplexities = []
        
        m = decoder_model.to(device)
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
                
            xb, yb = xb.to(device), yb.to(device)
            loss, maps = decoder_model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            perplexity = compute_perplexity(decoder_model, train_LM_loader)
            trainPerplexities.append(perplexity)
            
            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    decoder_model.eval()
                    train_perplexity = compute_perplexity(decoder_model, train_LM_loader)
                    decoder_model.train()
                    print(f"Iteration {i + 1}, Loss: {loss.item():.4f}, Perplexity: {train_perplexity:.2f}")

        plt.figure()
        plt.plot(range(1, len(trainPerplexities) + 1), trainPerplexities, label="Training Perplexity")
        plt.title("Training Perplexity")
        plt.xlabel("Iteration")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.savefig("PerplexityGraph.png")



        test_LM_obama_dataset = LanguageModelingDataset(tokenizer, open('speechesdataset/test_LM_obama.txt', 'r', encoding='utf-8').read(), block_size)
        test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, open('speechesdataset/test_LM_wbush.txt', 'r', encoding='utf-8').read(), block_size)
        test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, open('speechesdataset/test_LM_hbush.txt', 'r', encoding='utf-8').read(), block_size)
        test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size)
        test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size)
        test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size)

        test_perplexity_obama = compute_perplexity(decoder_model, test_LM_obama_loader)
        print(f"Perplexity on test_LM_obama.txt: {test_perplexity_obama:.2f}")

        test_perplexity_wbush = compute_perplexity(decoder_model, test_LM_wbush_loader)
        print(f"Perplexity on test_LM_wbush.txt: {test_perplexity_wbush:.2f}")

        test_perplexity_hbush = compute_perplexity(decoder_model, test_LM_hbush_loader)
        print(f"Perplexity on test_LM_hbush.txt: {test_perplexity_hbush:.2f}")
        
        #Sanity Check
        sentence = "This PA was so difficult."
        utilities = Utilities(tokenizer, decoder_model)
        utilities.sanity_check(sentence, block_size)



if __name__ == "__main__":
    main()
