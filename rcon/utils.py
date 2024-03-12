import time
import datetime
import random
import os
import csv
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import matthews_corrcoef

batch_size = 32
seed_val = 23


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cd = torch.cuda.current_device();
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(cd))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


def read_data(filename):
    """Returns labeled and unlabeled sentences as two dataframes"""

    df = pd.read_excel(filename, sheet_name=1, header=1, usecols=[2,3,4])
    print("Loaded {} sentences.".format(len(df)))

    df.rename(columns={'Match': 'ID', 'MatchExtended': 'text', 'Complete?': 'label'}, inplace=True)

    #Clean up the sentences which start with the requirement ID.
    df['text'] = df['text'].str.replace(r'^K\d{3,}', '', regex=True)

    df.fillna('')

    labeled = df[df['label'].isin([0.0, 1.0])]
    unlabeled = df[~df['label'].isin([0.0, 1.0])]

    print("Labeled sentences:   {}".format(len(labeled)))
    print("Unlabeled sentences: {}".format(len(unlabeled)))


    grouped = labeled.groupby('label')
    group_sizes = grouped.size()

    min_group_size = group_sizes.min()
    min_group_name = group_sizes.idxmin()

    print("Label {} with {} instances is the smaller group.".format(min_group_name, min_group_size))

    #Sample min_group_size from all groups
    labeled = grouped.apply(lambda x: x.sample(min_group_size))

    return labeled, unlabeled

def get_ids(df):
    return get_values(df, "ID", str)

def get_sentences(df):
    return get_values(df, "text", str)

def get_labels(df):
    return get_values(df, "label", int)

def get_values(df, column, type):
    return (df[column].values).astype(type)

def max_sentence_length(tokenizer, sentences):
    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    print("Max sentence length: {}".format(max_len))

    return max_len


def tokenize(modelname, sentences):
    """Returns input_ids and attention_masks"""
    tokenizer = AutoTokenizer.from_pretrained(modelname)

    max_len = max_sentence_length(tokenizer, sentences)

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens = True,
            max_length=max_len+1,
            truncation=True,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def create_dataset(input_ids, attention_masks):
    return TensorDataset(input_ids, attention_masks)

def create_datasets(input_ids, attention_masks, labels, test_p, train_p):
    """Returns training, validation and test data sets. test_p (0..1) is the holdout percentage to use for testing. train_p is the holdout percentage of the samples used for learning"""
    labels = torch.tensor(labels)

    dataset =  TensorDataset(input_ids, attention_masks, labels)

    total_size = len(dataset)
    test_size = total_size - int((1-test_p) * total_size)
    learn_size = total_size - test_size
    train_size = int(train_p * learn_size)
    val_size = learn_size - train_size

    assert test_size + learn_size == total_size
    assert test_size + train_size + val_size == total_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print('{:>5,} total samples'.format(total_size))
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    print('{:>5,} test samples'.format(test_size))

    return train_dataset, val_dataset, test_dataset

def create_dataloader(dataset, random_sampler):
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      sampler = RandomSampler(dataset) if random_sampler else SequentialSampler(dataset)
                      )

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(modelname, epochs, train_data, val_data):
    model = BertForSequenceClassification.from_pretrained(
        modelname,
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False
    )

    train_dataloader = create_dataloader(train_data, True)
    validation_dataloader = create.dataloader(val_data, False)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        eps=1e-8
    )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    device = get_device()

    training_stats = []

    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)

            loss = output.loss
            logits = output.logits
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                output = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)
                loss = output.loss
                logits = output.logits

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    print("")
    print("Training stats")

    pd.set_option('display.precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    return model

def save_model(model, directory):
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(directory, date_time)

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(path)
    #tokenizer.save_pretrained(path)

def load_model(directory):
    model = BertForSequenceClassification.from_pretrained(directory)
    #tokenizer = tokenizer.from_pretrained(output_dir)
    model.to(get_device())
    return model

def evaluate(model, test_data):
    print('Predicting labels for {:,} test sentences...'.format(len(test_data)))

    test_dataloader = create_dataloader(test_data, False)

    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    predictions, true_labels = [], []
    device = get_device()

    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    num_test_labels = len(test_data)
    num_positive_labels = 0

    for batch in test_data:
        num_positive_labels += batch[2]

    print('Positive samples: %d of %d (%.2f%%)' % (num_positive_labels,
                                                   num_test_labels,
                                                   (num_positive_labels / num_test_labels * 100.0)))

    matthews_set = []
    print("")
    print('Calculating Matthews Corr. Coef. for each batch...')
    for i in range(len(true_labels)):
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)


    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    print('Total MCC: %.3f' % mcc)

    flat_pred = np.argmax(np.concatenate(predictions, axis=0), axis=1)
    flat_true = np.concatenate(true_labels, axis=0)
    accuracy = np.sum(flat_pred == flat_true) / len(flat_pred)
    print('Accuracy: %3f' % accuracy)

def predict(model, dataset):
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    predictions = []

    unlabeled_dataloader = create_dataloader(dataset, False)
    device = get_device()
    t0 = time.time()

    for step, batch in enumerate(unlabeled_dataloader):
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(unlabeled_dataloader), elapsed))

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits to CPU
        logits = logits.detach().cpu().numpy()

        # Store predictions
        predictions.append(logits)

    print('    DONE.')

    return predictions

def save_predictions(predictions, ids, sentences, directory):
    flat_pred = np.argmax(np.concatenate(predictions, axis=0), axis=1)
    zipped = zip(ids, sentences, flat_pred)

    filename = "predictions_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    path = os.path.join(directory, filename)

    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in zipped:
            writer.writerow(row)

    print("Predictions have been stored in {}".format(path))
