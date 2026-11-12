# ЛР3. Безопасный CI/CD для MNIST

Рабочая ветка — `lab3`. ЛР1 и ЛР2 сохраняются в существующих ветках/метках.
Используется GitHub Actions; дополнительный сервер CI не требуется.

## Что изменено

`src/model.py` и формат NPZ из ЛР2 сохранены. Из `src/lab2.py` удалён только
неиспользуемый импорт `Path`, обнаруженный Ruff. Все 17 тестов ЛР2 сохранены.
Дополнительные тесты проверяют значения предобработки, метрику на известном
примере, загрузку и целостность тестового артефакта, границы model gate,
ошибки схемы данных и обработку отчёта сканера. На CI тест реальной модели
обязателен и не пропускается.

`requirements.txt` содержит только runtime: NumPy и Pillow. Исходный список
из ЛР2 сохранён как `requirements-lab2.txt`. `requirements-ci.in` задаёт версии
инструментов и библиотек исследования; `requirements-ci.lock` создаётся один
раз в чистом `.venv-ci` через pip freeze и сохраняется в Git. Python для CI
фиксируется файлом `ci/python-version`, базовый образ — digest в Dockerfile.

## Подготовка на Ubuntu

Все команды выполняются в `~/ml-project` под `mluser` в ветке `lab3`.
Docker должен быть установлен и доступен пользователю. Сначала:

```bash
git status
sha256sum -c reports/dataset.sha256
sha256sum -c metrics/model.sha256
docker info
```

Не переобучать модель, не удалять `metrics/baseline.json`, не менять эталоны,
чтобы заставить несовпадающий файл пройти проверку.

### 1. Чистое окружение проверок

```bash
python3 -m venv .venv-ci
source .venv-ci/bin/activate
python -m pip install pip==26.3
python -m pip install -r requirements-ci.in
python -m pip freeze > requirements-ci.lock
```

Не выполнять freeze из старого окружения Jupyter: оно содержит лишние пакеты.
Исходное `.venv` остаётся для демонстрации предыдущих лабораторных.

### 2. Однократная фиксация CI-эталона и шифрование модели

```bash
python -m src.ci init
cat ci/baseline.json
cat ci/data_policy.json
```

Команда проверяет, что используется именно модель baseline ЛР2, получает
выборку, оценивает её и создаёт `ci/baseline.json`, `ci/model_policy.json`,
`ci/data_policy.json` и `ci/model.enc`. Повторная инициализация запрещена,
чтобы не заменить эталон метриками текущего кандидата. Она не вызывается в CI.

Ключ Fernet создаётся в `~/.config/ml-lab3/model.key` с правами 600, вне Git.
Fernet обеспечивает шифрование и проверку подлинности зашифрованных данных.
В Git хранится только `ci/model.enc`, открытая модель остаётся в models/.
После расшифрования SHA256 сверяется с независимым эталоном из ЛР2.

### 3. Доступ GitHub Actions к модели

Нужен один секрет `MODEL_KEY`: он необходим для расшифрования модели,
которая по условиям ЛР2 не публикуется открыто. Других токенов для CI не нужно.
Добавить его можно без вывода значения в терминал через GitHub CLI:

```bash
sudo apt update
sudo apt install gh
gh auth login --hostname github.com --git-protocol ssh --web
gh secret set MODEL_KEY --repo Denich7/ml-security-mnist < ~/.config/ml-lab3/model.key
gh secret list --repo Denich7/ml-security-mnist
```

При предложении загрузить дополнительный SSH-ключ можно выбрать Skip,
если push уже работает. Авторизация CLI выполняется в браузере владельцем.
Не публиковать ключ, код авторизации и токены на скриншотах.

Секрет передаётся только шагу расшифрования. Внешние pull request
отклоняются до установки зависимостей и обращения к модели; для них
полноценная сборка не объявляется успешной. После ревью изменения можно
перенести в доверенную ветку. `pull_request_target` не используется.
Права `GITHUB_TOKEN` — только `contents: read`, сохранение credentials
checkout отключено. Для будущей командной работы следует ограничить право
изменения workflow и CI-эталонов через review/защиту веток.

### 4. Фиксация базового образа

```bash
docker pull python:3.14-slim-bookworm
image=$(docker image inspect python:3.14-slim-bookworm --format '{{index .RepoDigests 0}}')
docker run --rm --entrypoint python "$image" -c 'import platform; print(platform.python_version())' > ci/python-version
printf 'FROM %s\n' "$image" > Dockerfile
cat ci/Dockerfile.body >> Dockerfile
cat ci/python-version
head -1 Dockerfile
```

Плавающий тег используется только при начальном выборе; итоговый Dockerfile
содержит конкретный digest. Контейнер не содержит sklearn, Jupyter, секрет,
полный датасет и инструменты аудита. USER — `10001:10001`. Копируются только
модули инференса, NPZ, его эталон SHA256 и runtime requirements.
Smoke-проверка использует корректное по схеме чёрное изображение; проверяет
загрузку и предсказание, но не заменяет оценку точности на MNIST.

### 5. Сканер и полный локальный запуск

```bash
python tools/install_trivy.py
export PATH="$PWD/.tools:$PATH"
trivy --version
bash tools/check_lab3.sh
```

Сканер скачивается из официальных GitHub releases Aquasecurity, его версия
и SHA256 зафиксированы в `ci/trivy-lock.json`. Архив не распаковывается
произвольно: извлекаются байты только обычного файла `trivy`.
Полный скрипт прекращает выполнение при первой ошибке; `pipefail`
сохраняет отказ команд, вывод которых передаётся в tee.

## Контрольная выборка и baseline

Источник: официальный архив MNIST из ЛР1, SHA256
`731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
При отсутствии локального архива загрузка выполняется по фиксированному HTTPS
URL с лимитом 16 MiB, после чего проверяется хэш. Тесты сети не используют.

Выборка: первые 100 объектов каждого класса официального теста, порядок
классов 0–9, всего 1000 изображений. Случайная генерация не используется.
Это контроль повторяемости, а не новая независимая оценка обобщения модели.
Известный тест уже использовался в ЛР2; для итоговой независимой оценки
нужен отдельный ранее не использованный набор.

Схема NPZ: `images` — uint8 (1000,28,28), `labels` — uint8 (1000,),
`version` — строковый скаляр. Допустимы только эти три поля. Политика
отклоняет пропуски полей, неверные типы/формы, NaN/Inf, неверные метки,
недопустимый объём, долю класса вне 9–11% и изменения относительно хэшей.
Для uint8 NaN/Inf не представимы, а преобразование в float отклоняется.

Эталонные SHA256 считаются по каноническим NPY-байтам двух массивов
(включают форму и dtype); это исключает зависимость от метаданных ZIP внутри
NPZ. Значения хранятся в `ci/data_policy.json`, отдельно от данных. Изменять
данные вместе с политикой без ревью нельзя: хэш не защищает от атакующего,
который имеет право менять оба файла. Локальные архивы исключены из Git.

`metrics/baseline.json` ЛР2 не меняется. `ci/baseline.json` — та же модель,
однократно оценённая на CI-выборке. Кандидат оценивается на тех же объектах
через `predict` ЛР2 и ту же нормализацию float32 / 255. Основная метрика —
macro-F1 (sklearn, average=macro, zero_division=0).

Порог: `candidate >= baseline * 0.95`, включая равенство. Отдельная политика
в `ci/model_policy.json`. Отсутствующая метрика, строка, bool, NaN/Inf,
значение вне 0–1, несовместимые версия/хэш/объём данных и нулевой baseline
отклоняются. Baseline не перезаписывается в workflow.

## Pipeline и политика безопасности

Последовательность: checkout → фиксированные зависимости → расшифрование
и проверка модели → получение проверенного источника данных → Ruff → Bandit
→ pytest → pip-audit → data gate → оценка кандидата и model gate → Docker
build → Trivy config → Trivy image → проверка отчёта → non-root и smoke.
Последующие обязательные шаги выполняются только после успешных предыдущих.
Последний шаг `if: always()` сохраняет только reports/lab3 и не скрывает отказ.

Политика pip-audit: блокировать любые известные уязвимости; аудитируется
`requirements-ci.lock`, включающий runtime и все зависимости CI. Проверка
runtime отдельно: `pip-audit -r requirements.txt`.
Trivy image блокирует CRITICAL и UNKNOWN. Trivy config блокирует все
найденные проблемы Dockerfile. Ошибка загрузки базы, отсутствие/некорректность
отчёта или ошибка сканера завершают pipeline ошибкой. Исключений CVE нет.

Ruff проверяет E4/E7/E9/F. Единственное исключение E402 для `src/lab2.py`
обосновано настройкой matplotlib backend до импорта pyplot.
Bandit: точечное `nosec B310` только на вызове urlopen с жёстко заданным
официальным HTTPS-адресом; URL не поступает от пользователя, применяется
timeout, ограничение размера и проверка SHA256. Остальные находки блокируют CI.

Actions закреплены полными SHA коммитов официальных репозиториев.
Ограничение job — 25 минут. Отчёты доступны как Actions artifact 14 дней;
скачать их для сдачи до истечения срока. Дата проверки и commit записываются
в отчёт. Registry/deployment не используются, Docker socket не передаётся
контейнеру приложения. При запуске: network none, read-only, cap-drop ALL,
no-new-privileges, ограничения памяти, CPU и PID.

## Публикация и четыре демонстрации

До commit проверить `git diff` и `git status --short`. Ключ не должен
находиться в проекте; `data/`, `models/`, `.venv-ci/`, `.tools/` и отчёты CI
исключены. Добавить только перечисленные файлы:

```bash
git add .gitignore .dockerignore README.md LAB3.md requirements.txt requirements-lab2.txt requirements-ci.in requirements-ci.lock ruff.toml Dockerfile
git add src/ci.py src/smoke.py src/lab2.py tests/test_ci.py ci/ tools/ .github/workflows/ci.yml
git diff --cached --stat
git commit -m "Add secure ML CI, quality gates and non-root container"
git push -u origin lab3
```

На GitHub открыть Actions → Lab 3 - secure ML pipeline → запуск ветки lab3.
Получить успешный запуск до демонстраций. Если сканер выявил реальную
уязвимость, исправить dependency/base image, обновить соответствующий lock
и повторить; не ставить `continue-on-error` и не убирать gate.

Каждая демонстрация начинается от рабочей `lab3`, а не от предыдущего отказа:

```bash
git switch lab3
git switch -c lab3-demo-data
nano ci/scenario.json
```

Содержимое: `{"scenario": "data"}`. Сохранить Ctrl+O, Enter, Ctrl+X.

```bash
git add ci/scenario.json
git commit -m "Demo: reject malformed data copy"
git push -u origin lab3-demo-data
```

Аналогично создать `lab3-demo-model` от `lab3`, значение `"model"`;
`lab3-demo-security` от `lab3`, значение `"security"`. Отдельные коммиты и push.
Не объединять эти ветки в lab3. Закончив: `git switch lab3`.

| Сценарий | Нарушение | Ожидаемый отказ | Что не выполняется |
|---|---|---|---|
| normal | Нет | Успешная сборка и smoke | — |
| data | Копия images имеет форму N×28×27 | Data quality gate | Оценка, сборка, сканирование, smoke |
| model | Реальный константный предиктор на той же CI-выборке | Model quality gate | Сборка, сканирование, smoke |
| security | Тестовый отчёт с SIMULATED-CRITICAL после реального сканирования | Security report policy | Non-root/smoke, дальнейший выпуск |

Security-сценарий — имитация находки для проверки политики, не реальная
уязвимость образа. Реальные Trivy image/config выполняются и сохраняются
отдельно. Модельный сценарий — реальная оценка константного предиктора,
не искусственное изменение числа метрики.

## Что показать и сдать

1. GitHub ветка lab3: src/, tests/, ci/, Dockerfile, .dockerignore, workflow.
2. Один зелёный и три красных запуска Actions; на каждом видны ветка,
   commit, этап отказа и серые пропущенные последующие шаги.
3. CI artifacts: Ruff, Bandit, pytest, pip-audit, данные, кандидат,
   сравнение с baseline, Trivy image/config, время, commit, SHA256 модели.
4. Терминал: `docker run --rm --entrypoint id ml-project:lab3 -u` → 10001;
   `docker run --rm --network none ml-project:lab3` → status ok и prediction.
5. `ci/data_policy.json`, `ci/model_policy.json`, `ci/baseline.json`:
   объяснить хэши, распределение, macro-F1 и относительный порог 5%.

Скриншоты: (1) SHA256 OK и ветка; (2) тесты; (3) pip-audit/Bandit;
(4) data/model gate; (5) non-root/smoke; (6) Trivy; (7) зелёный Actions;
(8–10) три отдельных отказа с пропущенными шагами. Секреты не снимать.
Отчёт составляется по фактическим результатам запусков, а не по ожидаемым.
