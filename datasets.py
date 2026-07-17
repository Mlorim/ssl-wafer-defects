"""
datasets.py

Загрузка WM-811K, стратифицированный split labeled/unlabeled,
балансировка классов (SMOTE / undersampling), Dataset-классы и аугментации.

Ожидаемый формат исходных данных: WM-811K обычно распространяется как
LSWMD.pkl (pandas DataFrame с колонками 'waferMap', 'failureType', ...).
Если у вас другой формат — меняйте только load_wm811k, остальной код
работает с numpy-массивами (images, labels) и не зависит от формата файла.
"""

import os
import pickle
import random
import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def _patch_legacy_pandas_modules():
    """
    Старые версии LSWMD.pkl (WM-811K) были сериализованы pickle'ом очень
    старых pandas (< 0.20), где внутренние модули лежали по путям вида
    'pandas.indexes.base', 'pandas.indexes.numeric' и т.п. В современном
    pandas эти модули переехали в 'pandas.core.indexes.*', и pickle.load
    падает с ModuleNotFoundError, пытаясь импортировать старый путь.

    Регистрируем алиасы старых модулей в sys.modules, указывающие на
    актуальные модули pandas, чтобы pickle мог их найти. Безопасно —
    ничего не меняет в реальном pandas, только добавляет доп. ссылки
    в sys.modules на время распаковки файла.
    """
    legacy_to_current = {
        "pandas.indexes": "pandas.core.indexes.base",
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.numeric": "pandas.core.indexes.numeric",
        "pandas.indexes.range": "pandas.core.indexes.range",
        "pandas.indexes.multi": "pandas.core.indexes.multi",
        "pandas.core.index": "pandas.core.indexes.base",
    }
    import importlib

    for legacy_name, current_name in legacy_to_current.items():
        if legacy_name in sys.modules:
            continue
        try:
            current_module = importlib.import_module(current_name)
            sys.modules[legacy_name] = current_module
        except ImportError:
            # если и текущего пути нет (версия pandas отличается сильнее) —
            # просто пропускаем, pickle.load сам кинет понятную ошибку дальше
            continue

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomUnderSampler = None


# 9 классов дефектов WM-811K (порядок фиксирован для воспроизводимости метрик)
CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "None",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

WAFER_SIZE = 64  # ресайз всех wafer map до фиксированного размера (H=W=64)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resize_wafer(img: np.ndarray, size: int = WAFER_SIZE) -> np.ndarray:
    """Ресайз одной wafer map (2D array, значения 0/1/2) до size x size через nearest neighbor."""
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=(size, size), mode="nearest")
    return img.squeeze(0).squeeze(0).numpy()


def _parse_label(failure_type) -> Optional[str]:
    """
    Извлекает имя класса дефекта (одно из CLASS_NAMES, регистронезависимо) из
    поля failureType строки LSWMD.pkl. Возвращает None, если сэмпл unlabeled
    (пустой/отсутствующий failureType). Общая логика для load_wm811k и
    load_wm811k_native.
    """
    label_str = None
    if failure_type is not None:
        try:
            if isinstance(failure_type, (list, np.ndarray)):
                flat = np.array(failure_type).flatten()
                if len(flat) > 0 and flat[0] not in (None, ""):
                    label_str = str(flat[0])
            elif isinstance(failure_type, str) and failure_type != "":
                label_str = failure_type
        except Exception:
            label_str = None

    if label_str is None:
        return None

    label_lower = label_str.strip().lower()
    for cls_name in CLASS_NAMES:
        if cls_name.lower() == label_lower:
            return cls_name
    return None


def load_wm811k(
    path: str,
    only_labeled: bool = False,
    max_samples: Optional[int] = None,
    resize_to: int = WAFER_SIZE,
    one_hot: bool = False,
    pixel_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает WM-811K из .pkl файла (LSWMD.pkl формат).

    one_hot=False (по умолчанию, поведение не изменилось): images — grayscale
    [N, H, W] float32, нормализованные в pixel_range (по умолчанию [0, 1];
    методу MM-WAE статьи Zhang et al. нужен [-1, 1] — pixel_range=(-1.0, 1.0)).
    one_hot=True (нужно методу HybridCNN-ViT статьи SemiWaferNet): images —
    one-hot тензор [N, H, W, 3] (фон/годен/брак), без нормализации в pixel_range —
    one-hot кодируется ДО неё, чтобы не терять категориальность значений;
    ресайз всё равно nearest-neighbor (_resize_wafer), поэтому категории
    остаются целыми {0,1,2} и после ресайза.

    Returns:
        images: np.ndarray, см. выше
        labels: np.ndarray [N] int64, -1 для unlabeled образцов
        is_labeled: np.ndarray [N] bool
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Не найден файл датасета: {path}. "
            f"Скачайте LSWMD.pkl (WM-811K) и положите в data/raw/."
        )

    _patch_legacy_pandas_modules()
    with open(path, "rb") as f:
        # LSWMD.pkl часто сериализован под Python 2, где str == bytes.
        # encoding="latin1" - стандартный безопасный способ прочитать такой
        # pickle в Python 3 (byte-strings декодируются 1:1, без потери данных
        # для ASCII-совместимого содержимого вроде названий классов дефектов).
        df = pickle.load(f, encoding="latin1")

    images_list = []
    labels_list = []
    is_labeled_list = []

    for _, row in df.iterrows():
        wafer_map = row["waferMap"]
        failure_type = row.get("failureType", None)

        # failureType в LSWMD.pkl хранится как вложенный numpy array, иногда пустой.
        # Пустой массив = unlabeled сэмпл (нет данных вообще).
        # Строка 'none' (в нижнем регистре) = размеченный класс "нет дефекта",
        # это НЕ то же самое, что unlabeled, и должно маппиться в класс "None".
        label_key = _parse_label(failure_type)

        if label_key is not None:
            label = CLASS_TO_IDX[label_key]
            labeled = True
        else:
            if only_labeled:
                continue
            label = -1
            labeled = False

        resized = _resize_wafer(np.array(wafer_map, dtype=np.float32), size=resize_to)

        if one_hot:
            # категории {0: фон, 1: годен, 2: брак} сохраняются точно после
            # nearest-neighbor ресайза — кодируем ДО какой-либо нормализации
            image = np.stack([(resized == v).astype(np.float32) for v in (0, 1, 2)], axis=-1)
        else:
            # значения wafer map обычно {0: no chip, 1: normal die, 2: defect die}
            # нормализуем в [0, 1], затем при необходимости растягиваем в pixel_range
            max_val = resized.max()
            if max_val > 0:
                resized = resized / max_val
            lo, hi = pixel_range
            if (lo, hi) != (0.0, 1.0):
                resized = resized * (hi - lo) + lo
            image = resized

        images_list.append(image)
        labels_list.append(label)
        is_labeled_list.append(labeled)

        if max_samples is not None and len(images_list) >= max_samples:
            break

    images = np.stack(images_list).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    is_labeled = np.array(is_labeled_list, dtype=bool)

    return images, labels, is_labeled


def split_labeled_unlabeled(
    images: np.ndarray,
    labels: np.ndarray,
    labeled_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Стратифицированный split: labeled_ratio доли данных (по каждому классу
    поровну) остаются с лейблами, остальное помечается как unlabeled (label=-1).

    Соответствует настройке из статьи: "training was conducted on 10% of the
    WM811K dataset, with the remaining data serving as unlabeled input".

    Returns:
        images (не меняется), labels_split (с -1 для unlabeled), is_labeled_mask
    """
    rng = np.random.RandomState(seed)
    labels_split = labels.copy()
    is_labeled_mask = np.zeros(len(labels), dtype=bool)

    for cls in range(NUM_CLASSES):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_labeled = max(1, int(len(cls_idx) * labeled_ratio))
        labeled_idx = cls_idx[:n_labeled]
        is_labeled_mask[labeled_idx] = True

    labels_split[~is_labeled_mask] = -1

    return images, labels_split, is_labeled_mask


def _hybrid_balance(
    images: np.ndarray,
    labels: np.ndarray,
    target_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    "Hybrid sampling" из статьи SemiWaferNet: downsampling мажоритарных
    классов (в основном "None") до target_per_class, затем SMOTE-oversampling
    оставшихся миноритарных классов до того же target_per_class.
    """
    if SMOTE is None:
        raise ImportError(
            "imbalanced-learn не установлен. Установите: pip install imbalanced-learn"
        )

    rng = np.random.RandomState(seed)

    keep_idx = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) > target_per_class:
            cls_idx = rng.choice(cls_idx, size=target_per_class, replace=False)
        keep_idx.append(cls_idx)
    keep_idx = np.concatenate(keep_idx)
    images_ds, labels_ds = images[keep_idx], labels[keep_idx]

    counts = np.bincount(labels_ds, minlength=NUM_CLASSES)
    minority_classes = [c for c in range(NUM_CLASSES) if 0 < counts[c] < target_per_class]
    if not minority_classes:
        return images_ds, labels_ds

    n = len(images_ds)
    flat = images_ds.reshape(n, -1)

    min_class_size = min(counts[c] for c in minority_classes)
    k_neighbors = max(1, min(5, int(min_class_size) - 1))
    sampling_strategy = {cls: target_per_class for cls in minority_classes}
    sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)

    flat_resampled, labels_resampled = sampler.fit_resample(flat, labels_ds)
    images_resampled = flat_resampled.reshape((-1,) + images_ds.shape[1:]).astype(np.float32)
    return images_resampled, labels_resampled


def balance_classes(
    images: np.ndarray,
    labels: np.ndarray,
    method: str = "smote",
    seed: int = 42,
    target_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Балансировка labeled-части датасета.

    method:
        "smote"       - SMOTE oversampling минорных классов
        "undersample" - RandomUnderSampler majority классов
        "hybrid"      - downsampling мажоритарных + SMOTE миноритарных до
                         target_per_class (статья SemiWaferNet); требует target_per_class
        "none"        - без изменений
    """
    if method == "none":
        return images, labels

    if method == "hybrid":
        return _hybrid_balance(images, labels, target_per_class=target_per_class or 2000, seed=seed)

    if SMOTE is None:
        raise ImportError(
            "imbalanced-learn не установлен. Установите: pip install imbalanced-learn"
        )

    n = images.shape[0]
    flat = images.reshape(n, -1)

    if method == "smote":
        # для очень маленьких классов SMOTE требует k_neighbors < размера класса
        min_class_size = np.min(np.bincount(labels))
        k_neighbors = max(1, min(5, min_class_size - 1))
        sampler = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=seed)
    else:
        raise ValueError(f"Неизвестный метод балансировки: {method}")

    flat_resampled, labels_resampled = sampler.fit_resample(flat, labels)
    images_resampled = flat_resampled.reshape((-1,) + images.shape[1:]).astype(np.float32)

    return images_resampled, labels_resampled


class _WeakTransform:
    """Лёгкие аугментации (для teacher / labeled classification)."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
        if random.random() < 0.5:
            img = np.flipud(img).copy()
        k = random.choice([0, 1, 2, 3])
        img = np.rot90(img, k=k).copy()
        return img


class _StrongTransform:
    """
    Более агрессивные аугментации + шум (для student на unlabeled,
    используется как "Noise'" на Fig.1 статьи).
    """

    def __init__(self):
        self._weak = _WeakTransform()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = self._weak(img)
        # аддитивный гауссовский шум ("Noise" на Fig.1)
        noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
        # случайное затемнение небольшой области (аналог cutout)
        if random.random() < 0.3:
            h, w = img.shape
            ch, cw = h // 6, w // 6
            cy = random.randint(0, h - ch)
            cx = random.randint(0, w - cw)
            img[cy : cy + ch, cx : cx + cw] = 0
        return img


class _NoneTransform:
    """Без аугментаций (eval)."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img


def get_transforms(mode: str = "weak"):
    """
    Аугментации для wafer map (grayscale, без цветовых искажений).

    mode:
        "weak"   - лёгкие аугментации (для teacher / labeled classification)
        "strong" - более агрессивные аугментации + шум (для student на unlabeled,
                   используется как "Noise'" на Fig.1 статьи)
        "none"   - без аугментаций (eval)

    Возвращает callable-объект класса (не вложенную функцию), чтобы
    DataLoader с num_workers > 0 мог его запиклить при передаче в
    дочерние процессы (multiprocessing spawn на macOS/Windows требует
    pickle-совместимые объекты; локальные замыкания для этого не годятся).
    """
    if mode == "weak":
        return _WeakTransform()
    elif mode == "strong":
        return _StrongTransform()
    elif mode == "none":
        return _NoneTransform()
    else:
        raise ValueError(f"Неизвестный transform mode: {mode}")


class WaferDataset(Dataset):
    """
    Базовый датасет для labeled данных (или eval).
    Возвращает (image_tensor[1,H,W], label).
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform if transform is not None else get_transforms("none")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        label = int(self.labels[idx])
        return img_tensor, label


class UnlabeledWaferDataset(Dataset):
    """
    Датасет для unlabeled данных под Mean Teacher / consistency-методы.
    Возвращает пару (weak_view, strong_view) одного и того же изображения:
        weak_view  -> подаётся в teacher
        strong_view -> подаётся в student
    """

    def __init__(self, images: np.ndarray):
        self.images = images
        self.weak_transform = get_transforms("weak")
        self.strong_transform = get_transforms("strong")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        weak_img = self.weak_transform(img)
        strong_img = self.strong_transform(img)
        weak_tensor = torch.tensor(weak_img, dtype=torch.float32).unsqueeze(0)
        strong_tensor = torch.tensor(strong_img, dtype=torch.float32).unsqueeze(0)
        return weak_tensor, strong_tensor


class ContrastiveWaferDataset(Dataset):
    """
    Датасет для SupCon: возвращает labeled картинку в двух разных
    аугментированных версиях + лейбл (нужно 2 view на сэмпл для positive pairs).
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels
        self.transform = get_transforms("strong")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        view1_t = torch.tensor(view1, dtype=torch.float32).unsqueeze(0)
        view2_t = torch.tensor(view2, dtype=torch.float32).unsqueeze(0)
        label = int(self.labels[idx])
        return view1_t, view2_t, label


class WaferOneHotDataset(Dataset):
    """
    Аналог WaferDataset, но для one-hot тензоров [H, W, 3] (фон/годен/брак),
    используемых в методе CBAM-CNN (нативное разрешение, без ресайза).
    Возвращает (image_tensor[3, H, W], label). Аугментации из get_transforms
    (fliplr/flipud/rot90/noise/cutout) работают поэлементно и на 3D-массивах
    без изменений.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.transform = transform if transform is not None else get_transforms("none")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]
        label = int(self.labels[idx])
        return img_tensor, label


class WaferOneHotUnlabeledDataset(Dataset):
    """
    Unlabeled one-hot [H, W, 3] данные без аугментаций и без лейбла —
    используется HybridCNNViTMethod для MC-Dropout инференса при
    псевдо-разметке (нужен детерминированный порядок и исходное изображение).
    """

    def __init__(self, images: np.ndarray):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)


class VAEDataset(Dataset):
    """
    Обёртка для прогонов, где не нужен label: VAE-претрейн на всём пуле
    изображений (labeled + unlabeled) и teacher-инференс для псевдоразметки
    unlabeled данных (LatentVectorMethod). Без аугментаций — только исходное
    изображение, т.к. VAE учится реконструировать исходное распределение,
    а не инвариантность к аугментациям.
    """

    def __init__(self, images: np.ndarray):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


class DataLoaderFactory:
    """
    Тонкая обёртка вокруг (batch_size, num_workers), строит DataLoader любого
    из существующих Dataset-классов по требованию из numpy-массивов.

    Методы train.py больше не строят DataLoader'ы напрямую — вместо этого
    каждый метод декларирует, какие данные ему нужны (requires_unlabeled и
    т.п., см. methods/base.py), и вызывает нужные фабричные методы отсюда.
    Это также позволяет методам с несколькими фазами обучения (например
    LatentVectorMethod) перестраивать loader'ы между фазами — скажем, после
    псевдоразметки, когда набор "размеченных" данных меняется.
    """

    def __init__(self, batch_size: int, num_workers: int):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def labeled_loader(self, images, labels, transform_mode="weak", shuffle=True, drop_last=True, batch_size=None):
        dataset = WaferDataset(images, labels, transform=get_transforms(transform_mode))
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, drop_last=drop_last,
        )

    def contrastive_loader(self, images, labels, shuffle=True, drop_last=True, batch_size=None):
        dataset = ContrastiveWaferDataset(images, labels)
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, drop_last=drop_last,
        )

    def unlabeled_loader(self, images, shuffle=True, drop_last=True, batch_size=None):
        dataset = UnlabeledWaferDataset(images)
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, drop_last=drop_last,
        )

    def raw_pool_loader(self, images, shuffle=True, batch_size=None):
        """Весь image pool без лейблов — для VAE pretrain (LatentVectorMethod)."""
        dataset = VAEDataset(images)
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, drop_last=False,
        )

    def unlabeled_eval_loader(self, images, batch_size=None):
        """
        Детерминированный (shuffle=False, drop_last=False) проход по unlabeled
        данным без лейблов — для teacher-инференса при псевдоразметке.
        """
        dataset = VAEDataset(images)
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=False,
        )

    def eval_loader(self, images, labels, batch_size=None):
        dataset = WaferDataset(images, labels, transform=get_transforms("none"))
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )

    def onehot_loader(self, images, labels, transform_mode="weak", shuffle=True, drop_last=True, batch_size=None):
        """Labeled loader для one-hot [H,W,3] данных (CBAM-CNN)."""
        dataset = WaferOneHotDataset(images, labels, transform=get_transforms(transform_mode))
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, drop_last=drop_last,
        )

    def onehot_eval_loader(self, images, labels, batch_size=None):
        dataset = WaferOneHotDataset(images, labels, transform=get_transforms("none"))
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )

    def onehot_unlabeled_loader(self, images, batch_size=None):
        """Детерминированный (shuffle=False) проход по unlabeled one-hot данным без лейблов."""
        dataset = WaferOneHotUnlabeledDataset(images)
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=False,
        )


def load_wm811k_native(path: str, size: int = 26) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает WM-811K, оставляя только LABELED карты в исходном разрешении
    size x size (без ресайза — метод CBAM-CNN явно отбирает только карты
    нативного разрешения, чтобы избежать артефактов интерполяции), и кодирует
    каждую в one-hot тензор [size, size, 3] (фон/годен/брак). Метод полностью
    supervised — unlabeled карты не нужны и не загружаются.

    Returns:
        images: np.ndarray [N, size, size, 3] float32, {0, 1}
        labels: np.ndarray [N] int64
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Не найден файл датасета: {path}. "
            f"Скачайте LSWMD.pkl (WM-811K) и положите в data/raw/."
        )

    _patch_legacy_pandas_modules()
    with open(path, "rb") as f:
        df = pickle.load(f, encoding="latin1")

    images_list, labels_list = [], []
    for _, row in df.iterrows():
        wafer_map = np.array(row["waferMap"])
        if wafer_map.shape != (size, size):
            continue

        label_key = _parse_label(row.get("failureType", None))
        if label_key is None:
            continue

        onehot = np.stack([(wafer_map == c).astype(np.float32) for c in (0, 1, 2)], axis=-1)
        images_list.append(onehot)
        labels_list.append(CLASS_TO_IDX[label_key])

    if not images_list:
        raise ValueError(
            f"Не найдено ни одной размеченной карты нативного разрешения {size}x{size} в {path}"
        )

    images = np.stack(images_list).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    return images, labels


def _balance_via_cae(
    images: np.ndarray,
    labels: np.ndarray,
    target_per_class: int,
    cae_epochs: int,
    cae_noise_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Балансировка train-части для CBAM-CNN: мажоритарные классы (в основном
    "None") даунсэмплятся до target_per_class, миноритарные дополняются
    синтетическими образцами через CAE (обучен на всём train-наборе):
    Z = encode(X), Z_noisy = Z + eps (eps ~ N(0, cae_noise_std^2)),
    X_synth = decode(Z_noisy).

    Импорт models.WaferCAE — локальный (внутри функции), чтобы не вводить
    жёсткую top-level зависимость datasets.py -> models.py для остальных
    методов, которым CAE не нужен.
    """
    from models import WaferCAE  # локальный импорт, см. докстринг

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    cae = WaferCAE(input_channels=images.shape[-1], input_size=images.shape[1])
    optimizer = torch.optim.Adam(cae.parameters(), lr=1e-3)
    tensor_images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # [N,3,H,W]

    cae.train()
    n = len(tensor_images)
    batch_size = 64
    for _ in range(cae_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch = tensor_images[perm[i : i + batch_size]]
            recon = cae(batch)
            loss = F.mse_loss(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    cae.eval()

    out_images, out_labels = [], []
    with torch.no_grad():
        for cls in range(NUM_CLASSES):
            cls_idx = np.where(labels == cls)[0]
            n_cls = len(cls_idx)
            if n_cls == 0:
                continue

            if n_cls >= target_per_class:
                chosen = rng.choice(cls_idx, size=target_per_class, replace=False)
                out_images.append(images[chosen])
                out_labels.append(np.full(target_per_class, cls, dtype=np.int64))
            else:
                out_images.append(images[cls_idx])
                out_labels.append(labels[cls_idx])

                n_needed = target_per_class - n_cls
                src_idx = rng.choice(cls_idx, size=n_needed, replace=True)
                src_tensor = torch.tensor(images[src_idx], dtype=torch.float32).permute(0, 3, 1, 2)
                z = cae.encode(src_tensor)
                noise = torch.randn_like(z) * cae_noise_std
                synthetic = cae.decode(z + noise)
                synthetic_np = synthetic.permute(0, 2, 3, 1).numpy()

                out_images.append(synthetic_np)
                out_labels.append(np.full(n_needed, cls, dtype=np.int64))

    balanced_images = np.concatenate(out_images).astype(np.float32)
    balanced_labels = np.concatenate(out_labels).astype(np.int64)
    return balanced_images, balanced_labels


def prepare_datasets_cbam(
    data_path: str,
    native_size: int = 26,
    target_per_class: int = 2000,
    cae_epochs: int = 20,
    cae_noise_std: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Пайплайн подготовки данных для CBAM-CNN метода: фильтрация WM-811K до
    нативного разрешения native_size x native_size (без ресайза), one-hot
    encoding, honest train/val/test split (val_ratio доли — под early
    stopping/подбор оптимального числа эпох, val не участвует в CAE-аугментации
    и остаётся в исходном несбалансированном распределении, как test), затем
    балансировка train-части (downsampling "None" + CAE-аугментация
    миноритарных до target_per_class, см. _balance_via_cae).

    Возвращает dict с теми же ключами, что и prepare_datasets(), плюс
    "val_images"/"val_labels"; unlabeled_images — пустой массив, т.к. метод
    полностью supervised.
    """
    set_seed(seed)
    images, labels = load_wm811k_native(data_path, size=native_size)

    rng = np.random.RandomState(seed)
    n = len(images)
    perm = rng.permutation(n)
    test_size = int(n * 0.2)
    val_size = int(n * val_ratio)
    test_idx = perm[:test_size]
    val_idx = perm[test_size:test_size + val_size]
    train_idx = perm[test_size + val_size:]

    test_images, test_labels = images[test_idx], labels[test_idx]
    val_images, val_labels = images[val_idx], labels[val_idx]
    train_images, train_labels = images[train_idx], labels[train_idx]

    balanced_images, balanced_labels = _balance_via_cae(
        train_images, train_labels,
        target_per_class=target_per_class, cae_epochs=cae_epochs,
        cae_noise_std=cae_noise_std, seed=seed,
    )

    return {
        "labeled_images": balanced_images,
        "labeled_labels": balanced_labels,
        "unlabeled_images": np.empty((0, native_size, native_size, 3), dtype=np.float32),
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def prepare_datasets_hybrid_vit(
    data_path: str,
    resize_to: int = 32,
    unlabeled_pool_size: int = 150000,
    balance_target_per_class: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Пайплайн подготовки данных для HybridCNN-ViT (статья SemiWaferNet):
    one-hot [H,W,3] представление, ресайз до resize_to (по статье — 32x32).
    В отличие от prepare_datasets() (которая имитирует scarcity, искусственно
    скрывая лейблы у части полностью размеченного набора), здесь используется
    РЕАЛЬНАЯ граница labeled/unlabeled из WM-811K (is_labeled), а unlabeled пул
    подсэмплируется до unlabeled_pool_size (статья: 150 000). Балансировка
    train-части — "hybrid" (downsampling "None" + SMOTE миноритарных, formula
    из статьи), validation/test остаются в исходном (imbalanced) распределении.
    """
    set_seed(seed)
    images, labels, is_labeled = load_wm811k(data_path, only_labeled=False, resize_to=resize_to, one_hot=True)

    labeled_images_all = images[is_labeled]
    labeled_labels_all = labels[is_labeled]
    unlabeled_images_all = images[~is_labeled]

    rng = np.random.RandomState(seed)
    n = len(labeled_images_all)
    perm = rng.permutation(n)
    test_size = int(n * 0.2)
    test_idx, train_idx = perm[:test_size], perm[test_size:]

    test_images, test_labels = labeled_images_all[test_idx], labeled_labels_all[test_idx]
    train_images, train_labels = labeled_images_all[train_idx], labeled_labels_all[train_idx]

    balanced_images, balanced_labels = balance_classes(
        train_images, train_labels, method="hybrid", seed=seed, target_per_class=balance_target_per_class,
    )

    if len(unlabeled_images_all) > unlabeled_pool_size:
        chosen = rng.choice(len(unlabeled_images_all), size=unlabeled_pool_size, replace=False)
        unlabeled_images_all = unlabeled_images_all[chosen]

    return {
        "labeled_images": balanced_images,
        "labeled_labels": balanced_labels,
        "unlabeled_images": unlabeled_images_all,
        "test_images": test_images,
        "test_labels": test_labels,
    }


# Диспетчер подготовки данных по имени метода. Большинство методов используют
# общий grayscale-пайплайн prepare_datasets() (ресайз до 64x64, SMOTE/undersample).
# CBAM-CNN и HybridCNN-ViT требуют других пайплайнов (нативное разрешение /
# one-hot / hybrid-балансировка) — зарегистрированы здесь отдельно, чтобы не
# заводить if/elif в train.py.
_DATASET_PIPELINES = {
    "cbam_cnn": lambda config, seed: prepare_datasets_cbam(
        data_path=config["dataset"]["path"],
        native_size=config["dataset"].get("native_size", 26),
        target_per_class=config["dataset"].get("target_per_class", 2000),
        cae_epochs=config["dataset"].get("cae_epochs", 20),
        cae_noise_std=config["dataset"].get("cae_noise_std", 0.1),
        val_ratio=config["dataset"].get("val_ratio", 0.1),
        seed=seed,
    ),
    "hybrid_cnn_vit": lambda config, seed: prepare_datasets_hybrid_vit(
        data_path=config["dataset"]["path"],
        resize_to=config["dataset"].get("resize_to", 32),
        unlabeled_pool_size=config["dataset"].get("unlabeled_pool_size", 150000),
        balance_target_per_class=config["dataset"].get("balance_target_per_class", 2000),
        seed=seed,
    ),
    # MM-WAE (Zhang et al.): honest labeled_ratio-split из существующего
    # prepare_datasets() уже точно соответствует протоколу статьи (stratified
    # sampling 5%/10%, сохраняющий исходный imbalance, без oversampling —
    # имбаланс компенсируется class-frequency weighted CE самой модели), нужны
    # только другой resize (32x32) и диапазон пикселей ([-1,1] вместо [0,1]).
    "mm_wae": lambda config, seed: prepare_datasets(
        data_path=config["dataset"]["path"],
        labeled_ratio=config["dataset"].get("labeled_ratio", 0.10),
        balance_method=config["dataset"].get("balance_method", "none"),
        seed=seed,
        resize_to=config["dataset"].get("resize_to", 32),
        pixel_range=tuple(config["dataset"].get("pixel_range", [-1.0, 1.0])),
    ),
}


def prepare_data_for_method(method_name: str, config: dict, seed: int) -> dict:
    """Выбирает и запускает нужный пайплайн подготовки данных для метода."""
    pipeline = _DATASET_PIPELINES.get(method_name)
    if pipeline is not None:
        return pipeline(config, seed)

    return prepare_datasets(
        data_path=config["dataset"]["path"],
        labeled_ratio=config["dataset"]["labeled_ratio"],
        balance_method=config["dataset"]["balance_method"],
        seed=seed,
    )


def prepare_datasets(
    data_path: str,
    labeled_ratio: float = 0.10,
    balance_method: str = "smote",
    seed: int = 42,
    resize_to: int = WAFER_SIZE,
    pixel_range: Tuple[float, float] = (0.0, 1.0),
) -> dict:
    """
    Полный pipeline подготовки данных под обучение:
    load -> split labeled/unlabeled -> balance labeled часть.

    resize_to/pixel_range пробрасываются в load_wm811k — по умолчанию не
    меняют поведение (64x64, [0,1]); методу MM-WAE (Zhang et al.) нужны
    resize_to=32, pixel_range=(-1,1) — статья использует ИМЕННО этот honest
    train/test split + искусственно скрытые метки, что уже реализовано здесь,
    поэтому для MM-WAE не нужен отдельный prepare_datasets_* пайплайн.

    Returns dict с ключами:
        "labeled_images", "labeled_labels"       - сбалансированный labeled train
        "unlabeled_images"                        - unlabeled train (без лейблов)
        "test_images", "test_labels"              - test set (оставшиеся 90%, для честной оценки
                                                      используем held-out часть, не пересекающуюся
                                                      с unlabeled, см. примечание ниже)
    """
    set_seed(seed)
    images, labels, is_labeled = load_wm811k(data_path, only_labeled=True, resize_to=resize_to, pixel_range=pixel_range)

    # honest split: делим labeled-часть датасета на train/test, дальше из train
    # берём labeled_ratio как "размеченную" часть, остальное train уходит в unlabeled.
    # Это отличается от статьи (где test не описан явно) — фиксируем как наше допущение.
    rng = np.random.RandomState(seed)
    n = len(images)
    perm = rng.permutation(n)
    test_size = int(n * 0.2)
    test_idx, train_idx = perm[:test_size], perm[test_size:]

    test_images, test_labels = images[test_idx], labels[test_idx]
    train_images, train_labels = images[train_idx], labels[train_idx]

    _, train_labels_split, is_labeled_mask = split_labeled_unlabeled(
        train_images, train_labels, labeled_ratio=labeled_ratio, seed=seed
    )

    labeled_images = train_images[is_labeled_mask]
    labeled_labels = train_labels_split[is_labeled_mask]
    unlabeled_images = train_images[~is_labeled_mask]

    if balance_method != "none":
        labeled_images, labeled_labels = balance_classes(
            labeled_images, labeled_labels, method=balance_method, seed=seed
        )

    return {
        "labeled_images": labeled_images,
        "labeled_labels": labeled_labels,
        "unlabeled_images": unlabeled_images,
        "test_images": test_images,
        "test_labels": test_labels,
    }