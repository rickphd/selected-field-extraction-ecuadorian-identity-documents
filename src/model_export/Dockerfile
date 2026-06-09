# Imagen base de Ubuntu 22.04 con Python 3.11
FROM ubuntu:22.04

# Evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Configurar timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Actualizar sistema e instalar dependencias base
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Crear enlaces simbólicos para python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Crear directorio de trabajo
WORKDIR /app

# Actualizar pip
RUN python3 -m pip install --upgrade pip

# Copiar archivo de requerimientos
COPY requirements.txt .

# Instalar PyTorch
RUN pip install -r requirements.txt

# Aplicar fix de OpenCV
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

# Verificar instalaciones
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
RUN python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
RUN python -c "import onnx; print(f'ONNX version: {onnx.__version__}')"

# Configurar variables de entorno
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
