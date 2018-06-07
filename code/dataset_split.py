from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--label_csv_path', help='path to train.csv', default='/data/libiying/competition/competiton_final/train_all.csv')
parser.add_argument('--val_percentage',default=0.15)
opt = parser.parse_args()

def create_train_test(csv_path,val_percentage):
  frames = pd.read_csv(csv_path)
  row_idx_list = list(range(frames.shape[0]))
  train_idx_list, val_idx_list = train_test_split(row_idx_list, test_size=val_percentage) 
  return train_idx_list, val_idx_list

def main():
  with open("./train_list_all.csv",'a') as file_train:
    with open("./val_list_all.csv",'a') as file_val:
      train,val = create_train_test(opt.label_csv_path,opt.val_percentage) 
      writer_t = csv.writer(file_train)
      writer_v = csv.writer(file_val)
      for ft in train:
        writer_t.writerow([ft])
      for fv in val:
       writer_v.writerow([fv]) 


if __name__ == '__main__':
  main()
