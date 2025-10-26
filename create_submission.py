import zipfile
import os
from tqdm import tqdm

def create_zip_solutions():
    datasets = ['10', '20', '50', '140']
    
    print("Создание архивов...")
    
    for dataset in tqdm(datasets, desc="Архивирование наборов"):
        csv_file = f"solutions/solution_{dataset}.csv"
        zip_file = f"solutions/solution_{dataset}.zip"
        
        if os.path.exists(csv_file):
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(csv_file, os.path.basename(csv_file))
            
            print(f"  Создан архив: {zip_file}")
        else:
            print(f"  Файл {csv_file} не найден")

def create_single_zip_for_dataset(dataset):
    csv_file = f"solutions/solution_{dataset}.csv"
    zip_file = f"submission_{dataset}.zip"
    
    if not os.path.exists(csv_file):
        print(f"Файл {csv_file} не найден!")
        return False
    
    print(f"Создание архива для набора {dataset}...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file, f"solution_{dataset}.csv")
    
    print(f"Создан архив для отправки: {zip_file}")
    return True

def main():
    print("Создание архивов для всех наборов данных...")
    create_zip_solutions()
    
    print("\n" + "="*50)
    print("ИНСТРУКЦИЯ ДЛЯ ОТПРАВКИ:")
    print("="*50)
    print("Для отправки решения загрузите ОДИН из этих ZIP-файлов:")
    
    datasets = ['10', '20', '50', '140']
    for dataset in datasets:
        zip_file = f"solutions/solution_{dataset}.zip"
        if os.path.exists(zip_file):
            print(f" solutions/solution_{dataset}.zip")
        else:
            print(f" solutions/solution_{dataset}.zip (не найден)")
    
    print("\n Используйте: python create_submission.py --dataset 10")
    print("   для создания отдельного архива submission_10.zip")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Создать архив для конкретного набора (10, 20, 50, 140)')
    args = parser.parse_args()
    
    if args.dataset:
        if args.dataset in ['10', '20', '50', '140']:
            create_single_zip_for_dataset(args.dataset)
            print(f"\nЗагрузите файл submission_{args.dataset}.zip в систему")
        else:
            print("Неверный набор данных. Используйте: 10, 20, 50, 140")
    else:
        main()