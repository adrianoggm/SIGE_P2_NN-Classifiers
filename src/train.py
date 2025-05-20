import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from itertools import product
from config import DEVICE, EPOCHS
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb 
from src.customCNN import CustomCNN
import optuna
from optuna.trial import TrialState

def get_model(num_classes, model_type='resnet'):
    if model_type == 'resnet':
        # ResNet-18 pretrained
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    elif model_type == 'efficientnet_b4':
        # EfficientNet-B4 pretrained
        model = models.efficientnet_b4(pretrained=True)
        # Congelamos backbone
        for param in model.parameters():
            param.requires_grad = False
        # Sustituimos la cabeza (classifier)
        # En torchvision, classifier es [Dropout, Linear]
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    elif model_type == 'custom':
        from src.customCNN import CustomCNN
        model = CustomCNN(num_classes)

    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    return model.to(DEVICE)

def train_model(model,
                train_loader: DataLoader,
                val_loader: DataLoader,
                learning_rate: float,
                optimizer_name: str,
                save_best: bool = True,
                use_wandb: bool = False,
                with_htuning: bool = False):
    """
    Entrena el modelo y utiliza un scheduler ReduceLROnPlateau.
    Si el scheduler reduce el learning rate dos veces consecutivas sin mejora,
    detiene el entrenamiento.
    """

    if use_wandb and not with_htuning:
        wandb.init(
            project="clasification",
            config={
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,
                "epochs": EPOCHS
            },
            reinit=True 
        )
    criterion = nn.CrossEntropyLoss()

    # Configurar optimizador
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Scheduler: reduce LR on plateau (val_accuracy), con paciencia de 1 época
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',           # maximizamos val_accuracy
        factor=0.15,           # reducir LR al 10%
        patience=2,           # tras 1 época sin mejora
        verbose=True          # mostrar en consola
    )

    best_val_acc = 0.0
    lr_reduction_count = 0
    prev_lr = learning_rate

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_accuracy = correct / total * 100

        # Scheduler step basado en val_accuracy
        scheduler.step(val_accuracy)
        curr_lr = optimizer.param_groups[0]['lr']

        # Chequear mejora
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            lr_reduction_count = 0
            if save_best:
                torch.save(model.state_dict(), 'best_model.pth')
        else:
            # Si el LR se redujo, incrementar contador
            if curr_lr < prev_lr:
                lr_reduction_count += 1
                print(f"Learning rate reduced to {curr_lr:.6f} (count: {lr_reduction_count})")
            # Detener si dos reducciones consecutivas sin mejora
            if lr_reduction_count >= 2:
                print("No improvement after 2 LR reductions. Stopping training early.")
                break

        prev_lr = curr_lr

        # Logging y print
        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "lr": curr_lr
            })

    if use_wandb and not with_htuning:
        wandb.log({"final_val_accuracy": best_val_acc})
        wandb.finish()        

    return best_val_acc

def hyperparameter_tuning(train_dataset, val_dataset, full_dataset,
                          model_type='resnet'):

    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [32, 64],
        'optimizer': ['adam', 'sgd']
    }

    param_combinations = list(product(*param_grid.values()))
    num_classes = len(full_dataset.class_to_idx)

    best_config = None
    best_accuracy = 0.0

    for lr, batch_size, opt in param_combinations:
        print(f"\nProbando: lr={lr}, batch_size={batch_size}, optimizer={opt}, modelo={model_type}")

        wandb.init(
            project="clasification",
            config={
                "learning_rate": lr,
                "batch_size": batch_size,
                "optimizer": opt,
                "model_type": model_type,
                "epochs": EPOCHS
            },
            reinit=True 
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

        # Aquí pasamos model_type
        model = get_model(num_classes, model_type=model_type)
        val_acc = train_model(model,
                              train_loader, val_loader,
                              learning_rate=lr,
                              optimizer_name=opt,
                              use_wandb=True,
                              with_htuning=True)


        wandb.log({"final_val_accuracy": val_acc}) 
        wandb.finish() 
        
        print(f"Validación: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'optimizer': opt
            }

    print(f"\nMejor configuración encontrada para {model_type}: {best_config}")
    print(f"Precisión en validación: {best_accuracy:.2f}%")

    # 🏁 Entrenar modelo final con los mejores hiperparámetros
    best_train_loader = DataLoader(train_dataset,
                                   batch_size=best_config['batch_size'],
                                   shuffle=True)
    best_val_loader   = DataLoader(val_dataset,
                                   batch_size=best_config['batch_size'])
    final_model = get_model(num_classes, model_type=model_type)
    train_model(final_model,
                best_train_loader, best_val_loader,
                best_config['learning_rate'],
                best_config['optimizer'])

    # Guardar el modelo final
    wandb.init(project="clasification", name="final_model", config=best_config)
    train_model(final_model, best_train_loader, best_val_loader, best_config['learning_rate'], best_config['optimizer'])
    wandb.finish()

    return best_config





def objective(trial, train_dataset, val_dataset, num_classes, model_type='resnet'):
    torch.cuda.empty_cache()

    # Definir el espacio de búsqueda de hiperparámetros
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    
    # Para optimizadores más avanzados podríamos añadir:
    if optimizer_name == 'sgd':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
    
    # Para redes más complejas podríamos ajustar parámetros de arquitectura
    if model_type == 'custom':
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Configurar WandB para este trial
    wandb_config = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "model_type": model_type,
        "trial_number": trial.number
    }
    
    if optimizer_name == 'sgd':
        wandb_config["momentum"] = momentum
    if model_type == 'custom':
        wandb_config["dropout_rate"] = dropout_rate
    
    wandb.init(
        project="optuna-tuning-clasificacion",
        config=wandb_config,
        reinit=True,
        group=f"model_{model_type}",
        name=f"trial_{trial.number}"
    )

    # Crear data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Crear modelo
    model = get_model(num_classes, model_type=model_type)
    
    # Configurar optimizador
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        val_accuracy = correct / total * 100

        # Reportar métricas a Optuna y WandB
        trial.report(val_accuracy, epoch)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy
        })

        # Manejar pruning (podado) de trials que no van bien
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
    
        print(f'Epoch {epoch+1}/{EPOCHS} - Val Acc: {val_accuracy:.2f}%')

    wandb.finish()
    return best_val_acc

def hyperparameter_tuning_optuna(train_dataset, val_dataset, full_dataset, 
                                model_type='resnet', n_trials=20):
    
    num_classes = len(full_dataset.class_to_idx)
    
    # Función objetivo parcial para Optuna
    func = lambda trial: objective(trial, train_dataset, val_dataset, 
                                  num_classes, model_type)
    
    # Crear estudio Optuna
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Más trials completos para tener una buena mediana
            n_warmup_steps=5,    # Esperar hasta la época 10 para empezar a evaluar
            interval_steps=2      # Revisar cada 3 épocas (no todas)
        )
    )
    
    # Ejecutar optimización
    study.optimize(func, n_trials=n_trials, timeout=None)
    
    # Mostrar resultados
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nResultados del estudio:")
    print(f"Número de trials completados: {len(complete_trials)}")
    print(f"Número de trials podados: {len(pruned_trials)}")
    print(f"Mejor trial:")
    trial = study.best_trial
    print(f"  Valor (val_accuracy): {trial.value}")
    print("  Parámetros: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Entrenar modelo final con los mejores hiperparámetros
    best_params = trial.params
    print("\nEntrenando modelo final con los mejores hiperparámetros...")
    
    best_train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True
    )
    best_val_loader = DataLoader(
        val_dataset,
        batch_size=best_params['batch_size']
    )
    
    final_model = get_model(num_classes, model_type=model_type)
    
    # Configurar WandB para el modelo final
    wandb.init(
        project="optuna-tuning-clasificacion",
        name=f"final_model_{model_type}",
        config=best_params,
        group=f"model_{model_type}_final"
    )
    
    # Entrenar modelo final
    train_model(
        final_model,
        best_train_loader,
        best_val_loader,
        best_params['learning_rate'],
        best_params['optimizer'],
        use_wandb=True
    )
    
    # Guardar modelo
    torch.save(final_model.state_dict(), f'best_model_{model_type}.pth')
    wandb.save(f'best_model_{model_type}.pth')
    
    wandb.finish()
    
    return best_params