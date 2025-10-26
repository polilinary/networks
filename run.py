import os
import subprocess
import create_submission
import improved_solution
from tqdm import tqdm

def run_solution_for_dataset(dataset):
    """Запускает валидатор для конкретного набора данных"""
    solution_file = f"solutions/solution_{dataset}.csv"
    input_dir = os.path.join("data", dataset)
    
    print(f"\n{'='*60}")
    print(f"🔍 ПРОВЕРКА РЕШЕНИЯ ДЛЯ НАБОРА ДАННЫХ: {dataset}")
    print(f"{'='*60}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Папка с данными {input_dir} не найдена!")
        return False
    if not os.path.exists(solution_file):
        print(f"❌ Файл решения {solution_file} не найден!")
        return False
    
    cmd = [
        'python', 'validator.py',
        '-i', input_dir,
        '-s', solution_file
    ]
    
    print("  🧪 Запуск валидатора...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("❌ Ошибки:", result.stderr)
        return False
    
    return True

def main():
    # Создаем необходимые папки
    os.makedirs("solutions", exist_ok=True)
    
    print("🚚 Лабораторная работа: Оптимизация перевозок")
    print("📊 Обработка нескольких наборов данных")
    print("\n📁 Наборы данных: 10, 20, 50, 140")
    
    # Проверяем существование папок с данными
    datasets = ['10', '20', '50', '140']
    available_datasets = []
    
    print("\n🔎 Поиск наборов данных...")
    for dataset in tqdm(datasets, desc="Поиск данных"):
        data_path = os.path.join("data", dataset)
        if os.path.exists(data_path):
            available_datasets.append(dataset)
    
    # Выводим результаты поиска
    print("\n📋 Найдены наборы данных:")
    for dataset in datasets:
        if dataset in available_datasets:
            print(f"  ✅ {dataset}")
        else:
            print(f"  ❌ {dataset}")
    
    if not available_datasets:
        print("\n❌ Не найдено ни одного набора данных!")
        print("Убедитесь, что в папке 'data' есть подпапки: 10, 20, 50, 140")
        return
    
    input("\n⏎ Нажмите Enter чтобы начать обработку...")
    
    # Генерируем улучшенные решения для всех наборов
    print("\n🎯 Генерация улучшенных решений...")
    improved_solution.process_all_datasets()
    
    # Проверяем решения валидатором
    print("\n🔍 Проверка решений валидатором...")
    validation_results = {}
    
    for dataset in tqdm(available_datasets, desc="Валидация решений"):
        validation_results[dataset] = run_solution_for_dataset(dataset)
    
    # Создаем архивы для отправки
    print("\n📦 Создание архивов для отправки...")
    create_submission.main()
    
    # Итоговый отчет
    print("\n" + "="*60)
    print("✅ ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ!")
    print("="*60)
    
    print("\n📊 РЕЗУЛЬТАТЫ:")
    for dataset in available_datasets:
        status = "✅ УСПЕХ" if validation_results.get(dataset, False) else "❌ ОШИБКИ"
        print(f"  Набор {dataset}: {status}")
    
    print("\n📤 ДЛЯ ОТПРАВКИ В СИСТЕМУ:")
    print("Загрузите ОДИН из созданных ZIP-файлов:")
    for dataset in available_datasets:
        zip_file = f"solutions/solution_{dataset}.zip"
        if os.path.exists(zip_file):
            print(f"  ✅ solutions/solution_{dataset}.zip")
    
    print("\n💡 СОВЕТ: Начните с малого набора (10), затем переходите к larger")

if __name__ == "__main__":
    main()