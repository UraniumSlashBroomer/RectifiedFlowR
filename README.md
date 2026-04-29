# RectifiedFlowR
A repository where I implement Rectified Flow model from the [rectified flow paper](https://arxiv.org/pdf/2209.03003).


---

<img width="500" height="500" alt="results" src="https://github.com/user-attachments/assets/38d7e4a5-fb11-4b95-a8e5-f04ea837222f" />


---
## Language
- [Русский](#russian)

- [English](#english)

---
## Russian:

## Что это за репозиторий?

Тут я реализовываю rectified flow модель на pytorch с удобным интерфейсом для экспериментов. Обучаю модель на CIFAR10. В будущем я планирую добавить и другие небольшие датасеты.

---

### Как обучается rectified flow?

На вход rectified flow получает (X, t) и выдает векторное поле той же размерностью, что и X для момента времени $t \sim [0, 1]$.

Лосс модели следующий:

$$ L_{RF} = MSE(X_1 - X_0, V_\theta(X_t, t)), $$ 

где 

$X_1$ - изображение из датасета,

$X_0$ - шум,

$X_t = tX_1 + (1 - t)X_0$

$V_\theta(X_t, t)$ - предсказаное моделью векторное поле

---

### Что под капотом?

Под капотом я использую DiT (Diffusion Transformer). Для входа t в модель используется AdaLN (Adaptive Layer Norm).

Для генерации изображений я реализовал 3 ODE-Solver'а:

1. Euler
2. Heun
3. odeint (torchdiffeq lib) 

Формула для Euler ODE солвера:

$$ sample = sample + dt * model(sample, t_{curr}), $$

где

$t_{curr}$ - текущее t,

$dt$ - разница между $t_{curr}$ и предыдущим t,

само $t$ это linspace от 0 до 1 из T шагов.


Причем генерация происходит не на основных весах модели, а на EMA (Exponential Moving Average) весах. Таким образом генерации должны быть лучше, нежели если бы они были на весах основной модели.

Формула обновления EMA весов:

$$ \theta_{EMA} = decay * \theta_{EMA} + (1 - decay) * \theta_{model} $$

_decay обычно использут от 0.9 до 0.9999_

---

### Какие результаты я получил?

Изображения находятся в [results](#results). Надо отметить, что часть генераций на данный момент каша по нескольким причинам:

- Маленькое разрешение (32x32 у CIFAR10) на таком разрешении даже человеку иногда трудно понять, что изображено
- Отсутствие привязки к текстовым эмбеддингам. Планируется добавить в будущем

--- 

#### Дерево проекта:
```
- configs
- src
  - dataset
  - modules
    - modules.py
    - rectified_flow.py
  - utils
    - data_utils.py
    - initialization.py
    - utils.py
- train.py
- eval.py
```
---
#### Работа с проектом:

- Запуск обучения:
```
python train.py
```
|Flag|Type|Default|Description|
| :---  | :--- | :--- | :--- |
| `--device`  | `str` | `cuda` | Какой device использовать (cpu, cuda). |
| `--config`  | `str` | `.configs/default_config.yaml` | Путь к YAML конфиг файлу. |
| `--mode`  | `str` | `train` | Режим обучения (train, overfit, debug, train_c (продолжение обучения)), overfit и debug режимы используют их собственные конфиги (overfit_config, debug_config). train_c дает выбор эксперимента, какой вы хотите продолжить. |
| `--experiment` | `str` | None | Для явного указания эксперимента для продолжения обучения |
| `--wandb` | `bool` | из конфига | Подключать wandb или нет |
| `--batch_size`  | `int` | из конфига | Количество изображений в батче, переписывает значение из конфига |
| `--epochs`  | `int` | из конфига | Количество эпох, переписывает значение из конфига |
| `--num_training`  | `int` | из конфига | Количество изображений взятых из датасета для обучения, переписывает значение из конфига |
| `--decay` | `float` | из конфига | Decay для EMA модели, переписывает значение из конфига |
| `--warmup_epochs` | `int` | из конфига | Количество эпох для разогрева, переписывает значение из конфига |

Все, что запущено в debug моде, сохраняется в папку results/debug (она перезаписывается). Все остальное сохраняется в свои отдельные папки экспериментов.

- Тестирование модели:
```
python eval.py --device --solver['euler', 'heun', default='odeint']
```

После выбора эксперимента загружается модель и нужно ввести кол-во изображений для генерации и T если это требуется.

# English:

## What is this repository?

In this repo, I am implementing a Rectified Flow model using PyTorch, featuring a user-friendly interface for experimentation. The model trained on CIFAR10. I plan to add other small-size datasets in the feature.

---

### How is Rectified Flow trained?

Rectified Flow takes (X, t) as input and predicts a vector field of the same dimension as X for a time step $t \sim [0, 1]$.

The model's loss function is as follows:

$$ L_{RF} = MSE(X_1 - X_0, V_\theta(X_t, t)), $$ 

where:

$X_1$ - is an image from the dataset,

$X_0$ - is noise,

$X_t = tX_1 + (1 - t)X_0$

$V_\theta(X_t, t)$ - is the vector field predicted by the model.

---

### What’s under the hood?

Under the hood, I use a DiT (Diffusion Transformer) architecture. The time step is incorporated into the model using AdaLN (Adaptive Layer Norm).

For image generation, I have implemented three ODE Solvers:

1. Euler
2. Heun
3. odeint (torchdiffeq lib) 

The Euler ODE solver formula:

$$ sample = sample + dt * model(sample, t_{curr}), $$

where:

$t_{curr}$ - is the current time step,

$dt$ - is the difference between $t_{curr}$ and the previous time step t,

and $t$ itself is a linspace from 0 to 1 with T steps.


Furthermore, generation is performed using EMA (Exponential Moving Average) weights rather than the main model weights. This typically results in higher quality samples compared to using the raw model weights.

The EMA weight update formula:

$$ \theta_{EMA} = decay * \theta_{EMA} + (1 - decay) * \theta_{model} $$

_The decay value usually ranges from 0.9 to 0.9999_

---

### What results did I get?
The images can be found in the [results](#results) section. It should be noted that some of the generations currently look like "mush" for several reasons:

- Low resolution: CIFAR-10 uses 32x32 images. At this resolution, even for a human, it can sometimes be difficult to tell what is being depicted.
- Lack of text embeddings: The model is currently unconditional. I plan to add text-conditioning in the future.

---

#### How to use:

- Start training:
  
```
python train.py
```

|Flag|Type|Default|Description|
| :---  | :--- | :--- | :--- |
| `--device`  | `str` | `cuda` | Device to use (cuda, cpu). |
| `--config`  | `str` | `.configs/default_config.yaml` | Path to the YAML configuration file. |
| `--mode`  | `str` | `train` | Training mode (train, overfit, debug, train_c (continue training)), overfit and debug modes uses their own configs (overfit_config, debug_config). train_c gives you choose an experiment which you want to continue to train. |
| `--experiment` | `str` | None | specify a specific experiment to continue learning |
| `--wandb` | `bool` | from config | Is wandb linked or not |
| `--batch_size`  | `int` | from config | Number of samples per training step. Changes config number |
| `--epochs`  | `int` | from config | Number of epochs. Changes config number |
| `--num_training`  | `int` | from config | Number of training samples from the dataset (now only CIFAR10). Changes config number |
| `--decay` | `float` | from config | Decay for EMA weights. Changes config number |
| `--warmup_epochs` | `int` | from config | Number of warmup epochs. Changes config number |

Everything run in debug mode is saved to the results/debug folder (it will be overwritten). All other runs are saved in their respective experiment folders.

- Model Testing:
```
python eval.py --device --solver['euler', 'heun', default='odeint']
```

Once you have selected an experiment, the model will be loaded, and you will need to enter the number of images to generate and T (number of steps) if required.

---

# Results:
**Samples with odeint solver:**


<img width="500" height="500" alt="results" src="https://github.com/user-attachments/assets/9f90e474-09aa-44bb-aedc-079d75cbcedc" />


---
**Generation process with Heun solver (T = 50):**

<img width="500" height="500" alt="results" src="https://github.com/user-attachments/assets/df52d5dc-8619-4f1d-bce9-2024ec20c338" />


---
**Generation process with Heun solver (T = 25):**

<img width="1189" height="488" alt="results" src="https://github.com/user-attachments/assets/a6b655e5-aaa0-484d-8491-f1c91b9d0694" />

