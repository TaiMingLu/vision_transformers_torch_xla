# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

# WebDataset imports - install with: pip install webdataset
try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    # Only print warning once per process to avoid spam in multi-process training
    import os
    if os.environ.get('WEBDATASET_WARNING_SHOWN') != '1':
        print("WebDataset not available. Install with: pip install webdataset", flush=True)
        os.environ['WEBDATASET_WARNING_SHOWN'] = '1'

# FFCV imports - install with: pip install ffcv
try:
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage
    from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
    FFCV_AVAILABLE = True
except ImportError:
    FFCV_AVAILABLE = False
    # Suppressed to avoid multi-process spam: print("FFCV not available. Install with: pip install ffcv")

# HDF5 imports - install with: pip install h5py
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    # print("HDF5 not available. Install with: pip install h5py")

# psutil for RAM monitoring - install with: pip install psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Only print warning once per process to avoid spam in multi-process training
    import os
    if os.environ.get('PSUTIL_WARNING_SHOWN') != '1':
        print("psutil not available. Install with: pip install psutil (needed for RAM caching)", flush=True)
        os.environ['PSUTIL_WARNING_SHOWN'] = '1'

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET_WDS':  # WebDataset version of ImageNet
        print("reading WebDataset from datapath", args.data_path)
        dataset = build_webdataset_imagenet(args.data_path, is_train, transform, args)
        nb_classes = 1000
    elif args.data_set == 'IMNET_FFCV':  # FFCV version of ImageNet (FASTEST)
        print("reading FFCV from datapath", args.data_path)
        dataset = build_ffcv_imagenet(args.data_path, is_train, args)
        nb_classes = 1000
    elif args.data_set == 'IMNET_HDF5':  # HDF5 version of ImageNet (FAST & RELIABLE)
        print("reading HDF5 from datapath", args.data_path)
        dataset = build_hdf5_imagenet(args.data_path, is_train, transform, args)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes

class WebDatasetWithLength(torch.utils.data.Dataset):
    """Optimized WebDataset that behaves EXACTLY like ImageFolder with efficient TAR caching"""
    
    def __init__(self, sample_index, transform):
        super().__init__()
        self.sample_index = sample_index  # List of (tar_path, tar_member_name, class_idx)
        self.transform = transform
        self.length = len(sample_index)
        
        # Performance optimizations
        self._tar_cache = {}  # Cache open TAR files
        self._tar_cache_size = 10  # Keep up to 10 TAR files open
        self._tar_access_order = []  # LRU tracking
        
        import threading
        self._cache_lock = threading.Lock()  # Thread safety for DataLoader workers
    
    def _get_tar_file(self, tar_path):
        """Get TAR file with efficient caching (LRU eviction)"""
        import tarfile
        
        with self._cache_lock:
            # Check if already cached
            if tar_path in self._tar_cache:
                # Move to end (most recently used)
                self._tar_access_order.remove(tar_path)
                self._tar_access_order.append(tar_path)
                return self._tar_cache[tar_path]
            
            # Open new TAR file
            tar_file = tarfile.open(tar_path, 'r')
            
            # Add to cache
            self._tar_cache[tar_path] = tar_file
            self._tar_access_order.append(tar_path)
            
            # Evict oldest if cache is full
            if len(self._tar_cache) > self._tar_cache_size:
                oldest_path = self._tar_access_order.pop(0)
                oldest_tar = self._tar_cache.pop(oldest_path)
                try:
                    oldest_tar.close()
                except:
                    pass
            
            return tar_file
    
    def __getitem__(self, idx):
        from PIL import Image
        import io
        
        tar_path, img_member_name, class_idx = self.sample_index[idx]
        
        # Get cached TAR file (much faster than reopening)
        tar = self._get_tar_file(tar_path)
        
        try:
            # Extract image data
            img_data = tar.extractfile(img_member_name).read()
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # Apply transform (exactly like ImageFolder)
            if self.transform:
                image = self.transform(image)
            
            return image, class_idx
            
        except Exception as e:
            # Fallback: reopen TAR if cached version fails
            print(f"Warning: TAR cache miss for {tar_path}, reopening...")
            with self._cache_lock:
                if tar_path in self._tar_cache:
                    self._tar_cache.pop(tar_path)
                    if tar_path in self._tar_access_order:
                        self._tar_access_order.remove(tar_path)
            
            # Retry with fresh TAR file
            import tarfile
            with tarfile.open(tar_path, 'r') as fresh_tar:
                img_data = fresh_tar.extractfile(img_member_name).read()
                image = Image.open(io.BytesIO(img_data)).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                return image, class_idx
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        """Clean up cached TAR files"""
        try:
            with self._cache_lock:
                for tar_file in self._tar_cache.values():
                    try:
                        tar_file.close()
                    except:
                        pass
                self._tar_cache.clear()
        except:
            pass


def count_webdataset_samples(data_path, subset):
    """Count actual samples in WebDataset shards to get exact length"""
    import tarfile
    
    shard_dir = os.path.join(data_path, subset)
    if not os.path.exists(shard_dir):
        return 0
    
    total_samples = 0
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith('.tar')])
    
    print(f"Counting samples in {len(shard_files)} shards...")
    for shard_file in shard_files:
        shard_path = os.path.join(shard_dir, shard_file)
        try:
            with tarfile.open(shard_path, 'r') as tar:
                # Count image files (exclude .cls and .json files)
                image_files = [name for name in tar.getnames() 
                             if name.endswith(('.jpg', '.jpeg', '.png'))]
                total_samples += len(image_files)
        except Exception as e:
            print(f"Warning: Could not read {shard_file}: {e}")
            continue
    
    print(f"Found {total_samples} samples in {subset} split")
    return total_samples


def build_webdataset_index(data_path, subset):
    """Build optimized index of all samples in WebDataset TAR files"""
    import tarfile
    
    shard_dir = os.path.join(data_path, subset)
    if not os.path.exists(shard_dir):
        return []
    
    sample_index = []
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith('.tar')])
    
    print(f"Indexing {len(shard_files)} shards with performance optimizations...")
    
    for shard_file in shard_files:
        shard_path = os.path.join(shard_dir, shard_file)
        shard_samples = []  # Collect samples from this shard
        
        try:
            with tarfile.open(shard_path, 'r') as tar:
                # Get all members
                members = tar.getmembers()
                
                # Group by sample key
                sample_groups = {}
                for member in members:
                    if member.isfile():
                        # Extract sample key (before extension)
                        key = member.name.split('.')[0]
                        if key not in sample_groups:
                            sample_groups[key] = {}
                        
                        if member.name.endswith(('.jpg', '.jpeg', '.png')):
                            sample_groups[key]['image'] = member.name
                        elif member.name.endswith('.cls'):
                            sample_groups[key]['cls'] = member.name
                
                # Build index for each complete sample in this shard
                for key, files in sample_groups.items():
                    if 'image' in files and 'cls' in files:
                        try:
                            # Read class label to get class index
                            cls_data = tar.extractfile(files['cls']).read().decode('utf-8')
                            class_idx = int(cls_data.strip())
                            
                            # Store with shard grouping for better cache locality
                            shard_samples.append((shard_path, files['image'], class_idx))
                            
                        except Exception as e:
                            print(f"Warning: Could not index sample {key} in {shard_file}: {e}")
                            continue
                
                # Add all samples from this shard together (improves cache locality)
                sample_index.extend(shard_samples)
                            
        except Exception as e:
            print(f"Warning: Could not read {shard_file}: {e}")
            continue
    
    print(f"Indexed {len(sample_index)} samples with optimized TAR grouping")
    print(f"Performance optimization: samples grouped by TAR file for better caching")
    return sample_index


def build_webdataset_imagenet(data_path, is_train, transform, args):
    """
    Build WebDataset for ImageNet that behaves EXACTLY like ImageFolder
    
    Expected directory structure:
    data_path/
    ├── train/
    │   ├── imagenet-train-{000000..001281}.tar  # Training shards
    └── val/
        ├── imagenet-val-{000000..000049}.tar    # Validation shards
    """
    if not WEBDATASET_AVAILABLE:
        raise ImportError("WebDataset not available. Install with: pip install webdataset")
    
    subset = 'train' if is_train else 'val'
    
    # Build index of all samples (like ImageFolder builds file list)
    sample_index = build_webdataset_index(data_path, subset)
    
    # Create dataset that behaves EXACTLY like ImageFolder with on-demand loading
    return WebDatasetWithLength(sample_index, transform)


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_ffcv_imagenet(data_path, is_train, args):
    """
    Build FFCV loader for ImageNet - FASTEST option with exact ImageFolder behavior
    
    Expected FFCV files:
    data_path/
    ├── imagenet_high_quality_train.ffcv
    ├── imagenet_low_quality_train.ffcv  
    └── imagenet_val.ffcv
    """
    if not FFCV_AVAILABLE:
        raise ImportError("FFCV not available. Install with: pip install ffcv")
    
    # Determine which FFCV file to use
    if is_train:
        if 'high' in data_path.lower():
            ffcv_file = os.path.join(data_path, 'imagenet_high_quality_train.ffcv')
        elif 'low' in data_path.lower():
            ffcv_file = os.path.join(data_path, 'imagenet_low_quality_train.ffcv')
        else:
            # Default to high quality if not specified
            ffcv_file = os.path.join(data_path, 'imagenet_high_quality_train.ffcv')
    else:
        ffcv_file = os.path.join(data_path, 'imagenet_val.ffcv')
    
    if not os.path.exists(ffcv_file):
        raise FileNotFoundError(f"FFCV file not found: {ffcv_file}")
    
    print(f"Using FFCV file: {ffcv_file}")
    
    # Get normalization values
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    
    # Convert to [0, 255] range for FFCV
    mean_255 = [int(m * 255) for m in mean]
    std_255 = [int(s * 255) for s in std]
    
    # Define FFCV transforms (applied on GPU for speed)
    if is_train:
        # Training transforms - exact equivalent to build_transform
        image_pipeline = [
            RandomResizedCropRGBImageDecoder((args.input_size, args.input_size)),
            ToTensor(),
            ToDevice(torch.device('cuda'), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(mean_255, std_255, np.float32)
        ]
    else:
        # Validation transforms - exact equivalent to build_transform
        crop_ratio = getattr(args, 'crop_pct', None)
        if crop_ratio is None:
            crop_ratio = 224 / 256
        
        resize_size = int(args.input_size / crop_ratio)
        
        image_pipeline = [
            CenterCropRGBImageDecoder((args.input_size, args.input_size), ratio=crop_ratio),
            ToTensor(),
            ToDevice(torch.device('cuda'), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(mean_255, std_255, np.float32)
        ]
    
    # Label pipeline
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(torch.device('cuda'), non_blocking=True)
    ]
    
    # Create FFCV loader (this replaces DataLoader)
    loader = Loader(
        ffcv_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=OrderOption.RANDOM if is_train else OrderOption.SEQUENTIAL,
        drop_last=is_train,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline
        },
        distributed=torch.distributed.is_initialized()  # Auto-handle distributed training
    )
    
    return loader


class HDF5ImageDataset(torch.utils.data.Dataset):
    """ULTRA-FAST HDF5 dataset with optional RAM caching for GCS bucket optimization"""
    
    def __init__(self, hdf5_file, transform=None, cache_in_ram=False):
        super().__init__()
        
        if not HDF5_AVAILABLE:
            raise ImportError("HDF5 not available. Install with: pip install h5py")
        
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.cache_in_ram = cache_in_ram
        
        # Pre-import modules to avoid repeated imports in __getitem__
        from PIL import Image
        import io
        import time
        self._Image = Image
        self._io = io
        
        # Get dataset info first
        print(f"🔍 Opening HDF5 file to get dataset info: {hdf5_file}", flush=True)
        info_start = time.time()
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['images'])
            if cache_in_ram:
                # Get memory requirements
                images_shape = f['images'].shape
                labels_shape = f['labels'].shape
                images_dtype = f['images'].dtype
                labels_dtype = f['labels'].dtype
                
                # Estimate memory usage
                if images_dtype == 'object':
                    # Images stored as compressed JPEG bytes - use realistic ImageNet JPEG size
                    # Real ImageNet JPEG: 80-150KB per image, use 115KB as realistic estimate
                    avg_compressed_kb = 115  # Realistic estimate based on actual ImageNet
                    images_size_gb = (images_shape[0] * avg_compressed_kb) / (1024**2)
                    print(f"   📊 Estimated compressed JPEG size: ~{avg_compressed_kb} KB per image (realistic ImageNet)", flush=True)
                else:
                    # Raw pixel data
                    images_size_gb = (images_shape[0] * images_shape[1] if len(images_shape) > 1 else images_shape[0] * 1024) / (1024**3)
                
                labels_size_mb = (labels_shape[0] * 4) / (1024**2)  # Assume int32
                
                print(f"📊 Dataset info:", flush=True)
                print(f"   - Images: {images_shape}, dtype: {images_dtype}", flush=True)
                print(f"   - Labels: {labels_shape}, dtype: {labels_dtype}", flush=True)
                if images_dtype == 'object':
                    print(f"   - HDF5 storage size: ~{images_size_gb:.2f} GB + {labels_size_mb:.1f} MB (compressed)", flush=True)
                    decoded_size_gb = (images_shape[0] * 224 * 224 * 3) / (1024**3)  # Assume 224x224 RGB
                    print(f"   - Decoded RGB size: ~{decoded_size_gb:.1f} GB (when decompressed)", flush=True)
                else:
                    print(f"   - Estimated RAM usage: {images_size_gb:.2f} GB + {labels_size_mb:.1f} MB", flush=True)
        
        info_time = time.time() - info_start
        print(f"✅ Dataset info loaded in {info_time:.2f}s: {self.length} samples", flush=True)
        
        if cache_in_ram:
            self._load_all_to_ram()
        else:
            # For multi-worker compatibility, we'll open files in each worker
            self._h5_file = None
            self._images = None
            self._labels = None
            print(f"✅ HDF5 dataset configured for on-demand loading (multi-worker compatible)", flush=True)
    
    def _load_all_to_ram(self):
        """Load entire dataset to RAM for ultra-fast access (GCS bucket optimization)"""
        import time
        import os
        
        if not PSUTIL_AVAILABLE:
            print("⚠️  Warning: psutil not available, RAM monitoring disabled", flush=True)
            # Continue without RAM monitoring
            psutil = None
        else:
            import psutil
        
        print(f"🚀 Starting RAM cache loading from: {self.hdf5_file}", flush=True)
        
        # Check available RAM if psutil is available
        if psutil:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            print(f"💾 System RAM: {mem.total / (1024**3):.1f} GB total, {available_gb:.1f} GB available", flush=True)
        else:
            mem = None
        
        start_time = time.time()
        
        # Check if tqdm is available for progress bar
        try:
            from tqdm import tqdm
            tqdm_available = True
        except ImportError:
            if os.environ.get('TQDM_WARNING_SHOWN') != '1':
                print("Info: tqdm not available, progress bar disabled", flush=True)
                os.environ['TQDM_WARNING_SHOWN'] = '1'
            tqdm_available = False
        
        # Load everything in one shot for maximum efficiency
        print(f"📥 Opening HDF5 file for bulk loading...", flush=True)
        with h5py.File(self.hdf5_file, 'r') as f:
            # Get dataset info for progress setup
            images_shape = f['images'].shape
            labels_shape = f['labels'].shape
            
            # Simplified progress: Just show a spinner since HDF5 loading is atomic
            if tqdm_available:
                # Simple step-based progress (no byte estimates since HDF5 load is atomic)
                pbar = tqdm(total=2, desc="💾 Loading HDF5", unit="dataset", 
                           bar_format='{desc}: {n}/2 |{bar}| [{elapsed}, {postfix}]')
            
            print(f"📊 Loading images array ({images_shape[0]:,} samples, shape={images_shape})...", flush=True)
            load_start = time.time()
            
            # Load images - this is the big one
            self._cached_images = f['images'][:]
            images_time = time.time() - load_start
            images_size_gb = self._cached_images.nbytes / (1024**3)
            print(f"✅ Images loaded in {images_time:.1f}s: {images_size_gb:.2f} GB", flush=True)
            
            if tqdm_available:
                pbar.update(1)
                pbar.set_postfix_str(f"Images: {images_size_gb:.1f}GB in {images_time:.1f}s")
            
            # Load labels - should be fast
            print(f"📊 Loading labels array ({labels_shape[0]:,} samples, shape={labels_shape})...", flush=True)
            labels_start = time.time()
            self._cached_labels = f['labels'][:]
            labels_time = time.time() - labels_start
            labels_size_mb = self._cached_labels.nbytes / (1024**2)
            print(f"✅ Labels loaded in {labels_time:.3f}s: {labels_size_mb:.1f} MB", flush=True)
            
            if tqdm_available:
                pbar.update(1)
                pbar.set_postfix_str(f"Complete: {images_size_gb:.1f}GB + {labels_size_mb:.0f}MB")
                pbar.close()
        
        total_time = time.time() - start_time
        total_size_gb = (self._cached_images.nbytes + self._cached_labels.nbytes) / (1024**3)
        throughput_mbps = (total_size_gb * 1024) / total_time
        
        print(f"🎉 RAM cache loaded successfully!", flush=True)
        print(f"   - Total time: {total_time:.1f}s", flush=True)
        print(f"   - Total size: {total_size_gb:.2f} GB", flush=True)
        print(f"   - Throughput: {throughput_mbps:.1f} MB/s", flush=True)
        
        # Check final memory usage if psutil is available
        if psutil and mem:
            mem_after = psutil.virtual_memory()
            used_gb = (mem.available - mem_after.available) / (1024**3)
            print(f"💾 RAM usage increased by: {used_gb:.2f} GB", flush=True)
            print(f"💾 RAM available now: {mem_after.available / (1024**3):.1f} GB", flush=True)
    
    def _ensure_file_open(self):
        """Open HDF5 file if not already open (worker-safe)"""
        if self._h5_file is None:
            # CRITICAL: Open with optimal settings for ImageFolder-level performance
            self._h5_file = h5py.File(
                self.hdf5_file, 'r',
                rdcc_nbytes=1024*1024*128,    # 128MB cache per worker
                rdcc_nslots=10007,            # Prime number of cache slots
                rdcc_w0=0.75                  # Cache eviction policy
            )
            self._images = self._h5_file['images']
            self._labels = self._h5_file['labels']
    
    def __getitem__(self, idx):
        if self.cache_in_ram:
            # LIGHTNING-FAST: Direct RAM access - no network I/O!
            img_data = self._cached_images[idx]
            label = int(self._cached_labels[idx])
        else:
            # Ensure HDF5 file is open in this worker
            self._ensure_file_open()
            
            # Load image data efficiently (but from GCS bucket - slower)
            img_data = self._images[idx]
            
            # Load label directly (int conversion is cached by Python)
            label = int(self._labels[idx])
        
        # Decode image with minimal overhead (same as ImageFolder)
        image = self._Image.open(self._io.BytesIO(img_data)).convert('RGB')
        
        # Apply transform (exactly like ImageFolder)
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        """Clean up HDF5 file handle"""
        if hasattr(self, '_h5_file') and self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass


def build_hdf5_imagenet(data_path, is_train, transform, args):
    """
    Build HDF5 dataset for ImageNet - FAST and RELIABLE
    
    Expected directory structure:
    data_path/
    ├── train.h5   # Training data (quality determined by directory path)
    └── val.h5     # Validation data (same for all splits)
    
    Examples:
    - /path/to/imagenet-high-hdf5/ -> High quality split
    - /path/to/imagenet-low-hdf5/  -> Low quality split
    - /path/to/imagenet-custom/    -> Any custom split
    """
    if not HDF5_AVAILABLE:
        raise ImportError("HDF5 not available. Install with: pip install h5py")
    
    # Simple, clean filename structure
    if is_train:
        hdf5_file = os.path.join(data_path, 'train.h5')
    else:
        hdf5_file = os.path.join(data_path, 'val.h5')
    
    if not os.path.exists(hdf5_file):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")
    
    print(f"Using HDF5 file: {hdf5_file}")
    
    # Check if we should cache in RAM (TPU optimization for GCS bucket)
    cache_in_ram = getattr(args, 'cache_dataset_in_ram', False)
    if cache_in_ram:
        print(f"🚀 RAM caching enabled - will load entire dataset to memory for maximum speed")
    else:
        print(f"📁 On-demand loading enabled - will read from file for each sample")
    
    # Create dataset that behaves exactly like ImageFolder
    dataset = HDF5ImageDataset(hdf5_file, transform, cache_in_ram=cache_in_ram)
    
    return dataset
