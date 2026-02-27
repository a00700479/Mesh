import pandas as pd
import numpy as np
import os, math
import random
import datetime as dt
from collections import Counter, defaultdict

# ---- Paths ----
BASE_DIR = os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(BASE_DIR, "meal_planner_template_short.xlsx")

# ---- Unit normalization ----
UNIT_MAP = {
    "кг": ("г", 1000),
    "г": ("г", 1),
    "л": ("мл", 1000),
    "мл": ("мл", 1),
    "шт": ("шт", 1),
    "зуб": ("зуб", 1),
    "зубчик": ("зуб", 1),
    "ст.л.": ("ст.л.", 1),
    "ч.л.": ("ч.л.", 1),
}

# дополнительные варианты написания - добавляем к словарю
UNIT_MAP.update(
    {
        "ст. л.": ("ст.л.", 1),
        "ст л": ("ст.л.", 1),
        "ч. л.": ("ч.л.", 1),
        "ч л": ("ч.л.", 1),
        "гр": ("г", 1),
        "гр.": ("г", 1),
        "грамм": ("г", 1),
        "граммов": ("г", 1),
        "мл.": ("мл", 1),
        "лист": ("шт", 1),
        "листы": ("шт", 1),
    }
)
SPOON_TO_ML = {"ст.л.": 15, "ч.л.": 5}


def normalize_unit(unit):
    u = str(unit).strip().lower() if unit is not None else ""
    return UNIT_MAP.get(u, (u, 1))


def load_data(xlsx_path):
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    df_recipes = xl.parse("Recipes")

    # Ingredients
    df_ing = xl.parse("Ingredients")
    # нормализуем названия столбцов
    df_ing.columns = [str(c).strip() for c in df_ing.columns]
    # приводим варианты имени колонки к "Source"
    rename_map = {c: "Source" for c in df_ing.columns if c.strip().lower() == "source"}
    df_ing = df_ing.rename(columns=rename_map)
    print("Ingredient columns:", list(df_ing.columns))

    df_targets = (
        xl.parse("TagTargets")
        if "TagTargets" in xl.sheet_names
        else pd.DataFrame(columns=["Tag", "Target", "Min", "Max"])
    )
    df_cfg = (
        xl.parse("Config")
        if "Config" in xl.sheet_names
        else pd.DataFrame(columns=["Key", "Value"])
    )
    df_history = (
        xl.parse("History")
        if "History" in xl.sheet_names
        else pd.DataFrame(columns=["WeekStart", "Recipe", "Servings"])
    )
    df_pantry = (
        xl.parse("Pantry")
        if "Pantry" in xl.sheet_names
        else pd.DataFrame(columns=["Item", "QtyAvailable", "Unit"])
    )

    cfg = {
        str(k): str(v) for k, v in zip(df_cfg.get("Key", []), df_cfg.get("Value", []))
    }

    def _to_int(cfg, key, default):
        v = pd.to_numeric(cfg.get(key, None), errors="coerce")
        return int(default if pd.isna(v) else v)

    def _to_float(cfg, key, default):
        v = pd.to_numeric(cfg.get(key, None), errors="coerce")
        return float(default if pd.isna(v) else v)

    # числовые / булевые с дефолтами
    cfg["ServingsPerMeal"] = _to_int(cfg, "ServingsPerMeal", 2)
    cfg["NoRepeatWeeks"] = _to_int(cfg, "NoRepeatWeeks", 3)
    cfg["RecipesPerWeek"] = _to_int(cfg, "RecipesPerWeek", 7)
    cfg["ConvertSpoonsToMl"] = (
        str(cfg.get("ConvertSpoonsToMl", "False")).strip().lower() == "true"
    )
    cfg["Randomize"] = str(cfg.get("Randomize", "False")).strip().lower() == "true"
    cfg["RandomSeed"] = str(cfg.get("RandomSeed", "auto")).strip().lower()
    cfg["RandomJitter"] = _to_float(cfg, "RandomJitter", 0.5)
    cfg["RotationWeight"] = _to_float(cfg, "RotationWeight", 1.0)
    cfg["IngredientsBasis"] = (
        str(cfg.get("IngredientsBasis", "per_recipe")).strip().lower()
    )  # под мой текущий ввод на 2/4 порции

    # приведение типов History.WeekStart к датам (рано, но удобно)
    if not df_history.empty:
        df_history["WeekStart"] = pd.to_datetime(
            df_history["WeekStart"], errors="coerce"
        ).dt.date

    return df_recipes, df_ing, df_targets, cfg, df_history, df_pantry


def parse_tags(tags_str: str):
    # "рыба, китайское, емельяненко" -> ["рыба", "китайское", "емельяненко"]
    if tags_str is None:
        return []
    return [t.strip().lower() for t in str(tags_str).split(",") if t.strip()]


def compute_tag_ages(df_recipes, df_history, week_start):
    """
    tag -> weeks_since_last_seen.
    Теги считаем по текущему справочнику Recipes  (только теги существующих рецептов)
    Если тег никогда не встречался в History - вернем 52 (считаем "давно не было").

    """
    # рецепт -> теги (из справочника на сейчас)
    recipe_tags = {}
    for _, r in df_recipes.iterrows():
        recipe_tags[str(r["Recipe"]).strip().lower()] = parse_tags(r.get("Tags", ""))

    last_seen = {}  # tags -> last WeekStart date
    if df_history is not None and not df_history.empty:
        for _, row in df_history.iterrows():
            rname = str(row.get("Recipe", "")).strip().lower()
            ws = row.get(
                "WeekStart"
            )  # достаем значение из строки (row) по ключу "WeekStart"

            # если это уже date - используем напрямую
            if isinstance(ws, dt.date) and not isinstance(ws, dt.datetime):
                ws_date = ws
            # если это datetime - берем .date()
            elif isinstance(ws, dt.datetime):
                ws_date = ws.date()
            else:
                # на всякий случай мягко парсим строки/прочее
                ws_parsed = pd.to_datetime(ws, errors="coerce")
                if pd.isna(ws_parsed):
                    continue
                ws_date = ws_parsed.date()

            for t in recipe_tags.get(rname, []):
                prev = last_seen.get(t)
                if (prev is None) or (ws_date > prev):
                    last_seen[t] = ws_date
    ages = defaultdict(lambda: 52)  # по умолчанию считаем "давно не было"
    for t, d in last_seen.items():
        ages[t] = max(0, (week_start - d).days // 7)
    return ages


def choose_week(df_recipes, df_targets, df_history, cfg, week_start):
    """
    Возвращает DataFrame плана: [Recipe, Tags, Servings]
    Требования к данным:
    - df_recipes: колонки ["Recipe", "Tags", "DefaultServings", ....]
    - df_targets: колонки ["Tag", "Target", "Min", "Max"]
    - cfg: dict с ключами "RecipesPerWeek", "ServingsPerMeal", "NoRepeatWeeks", "Randomize", "RandomSeed", "RandomJitter", "RotationWeight"

    """
    # --- настройки ---
    need = int(cfg.get("RecipesPerWeek", 7))
    serv = int(cfg.get("ServingsPerMeal", 2))
    norep = int(cfg.get("NoRepeatWeeks", 3))
    rand_on = bool(cfg.get("Randomize", False))
    rand_seed = str(cfg.get("RandomSeed", "auto")).strip().lower()
    jitter = float(cfg.get("RandomJitter", 0.5))
    rot_w = float(cfg.get("RotationWeight", 1.0))

    # Инициализация генератора случайных чисел
    if rand_seed != "auto":
        try:
            random.seed(int(rand_seed))
        except Exception:
            random.seed()
    else:
        random.seed()

    # --- цели по тегам ---
    # TagTargets -> dict: tag -> (Target, Min, Max)
    targets = {}
    if df_targets is not None and not df_targets.empty:
        for _, row in df_targets.iterrows():
            tag = str(row.get("Tag", "")).strip().lower()
            if not tag:
                continue
            tgt = int(float(row.get("Target", 0) or 0))
            mn = int(float(row.get("Min", 0) or 0))
            mx_raw = row.get("Max", None)
            if (
                mx_raw is None
                or mx_raw == ""
                or (isinstance(mx_raw, float) and math.isnan(mx_raw))
            ):
                # Пустая ячейка Max -> считаем, что лимита нет
                mx = 10**6
            else:
                # Ячейка непустая -> используем значение как есть (0 остается 0, -1 остается -1)
                mx = int(float(mx_raw))
            targets[tag] = (tgt, mn, mx)

    # ---- Список "запрещенных" рецептов по истории последних norep недель ----
    # это рецепты, запрещенные по истории последних norep недель
    banned_recipes = set()
    if df_history is not None and not df_history.empty and norep > 0:
        # считаем границу
        cutoff = week_start - dt.timedelta(weeks=norep)
        for _, row in df_history.iterrows():
            ws = row.get("WeekStart")
            try:
                ws_date = (
                    ws
                    if isinstance(ws, dt.date)
                    else dt.datetime.strptime(str(ws), "%Y-%m-%d").date()
                )
            except Exception:
                continue
            if ws_date >= cutoff:  # включительно
                banned_recipes.add(str(row.get("Recipe", "")).strip().lower())

    # ---- "возраст" тегов (для ротации) ----
    tag_ages = compute_tag_ages(df_recipes, df_history, week_start)

    # --- формируем кандидатов ---
    # recs: список кортежей (score, name, tags_list, default_serv)
    recs = []
    for _, r in df_recipes.iterrows():
        nm = str(r["Recipe"]).strip()
        if not nm:
            continue
        name_key = nm.lower()
        if name_key in banned_recipes:
            # пока исключаем, но возможно потом используем для LRU-бекфилла - если захотим
            continue

        tags = parse_tags(r.get("Tags", ""))
        val = pd.to_numeric(r.get("DefaultServing", None), errors="coerce")
        default_serv = int(serv if pd.isna(val) else val)

        # базовый скор: теги, попадающие в TagTargets, чем больше тегов, попадающих в targets - тем лучше
        score = 0.0
        for t in tags:
            if t in targets:
                score += 2.0

        # бонус за разнообразие
        score += 0.2 * len(tags)

        # бонус ротации: чем дольше тег не попадал, тем выше
        score += rot_w * sum(tag_ages.get(t, 52) for t in tags)

        # мягкая случайность (джиттер)
        if rand_on:
            score += jitter * ((random.random() - 0.5) * 2.0)

        recs.append((score, nm, tags, default_serv))

    # случайная развязка ничьих
    if rand_on:
        random.shuffle(recs)
    # итоговая сортировка: по убыванию score, затем по имени
    recs.sort(key=lambda x: (-x[0], x[1]))

    # ---- набор плана с учетом Min/Target/Max ----
    plan = []
    tag_count = Counter()

    def already_in_plan(nm: str) -> bool:
        nk = nm.strip().lower()
        return any(p["Recipe"].strip().lower() == nk for p in plan)

    def can_add(tags_list) -> bool:
        # проверяем Max-ограничения (не превышаем)
        for t in tags_list:
            if t in targets:
                _, _, mx = targets[t]
                if tag_count[t] + 1 > mx:
                    return False
        return True

    def add_recipe(nm, tags_list):
        plan.append({"Recipe": nm, "Tags": ", ".join(tags_list), "Servings": serv})
        for t in tags_list:
            if t in targets:
                tag_count[t] += 1

    # 1) сначала закрываем Min (строгие минимумы) для каждого тега
    for tag, (_tgt, mn, _mx) in targets.items():
        while tag_count[tag] < mn and len(plan) < need:
            # ищем первого кандидата , у которого есть этот тег и которого можно добавить
            chosen = None
            for sc, nm, tags_list, _dsv in recs:
                if already_in_plan(nm):
                    continue
                if tag in tags_list and can_add(tags_list):
                    chosen = (nm, tags_list)
                    break
            if chosen is None:
                break  # не нашли - пойдем дальше, доберем потом
            add_recipe(*chosen)

    # 2) затем стараемся достичь Target
    for tag, (tgt, _mn, _mx) in targets.items():
        while tag_count[tag] < tgt and len(plan) < need:
            chosen = None
            for sc, nm, tags_list, _dsv in recs:
                if already_in_plan(nm):
                    continue
                if tag in tags_list and can_add(tags_list):
                    chosen = (nm, tags_list)
                    break
            if chosen is None:
                break
            add_recipe(*chosen)

    # 3) добираем оставшиеся слоты лучшими доступными кандидатами
    # ( с элементом случайности из верхнего пула)

    remaining = need - len(plan)
    if remaining > 0:
        pool = recs[: max(20, remaining * 3)] if rand_on else recs
        if rand_on:
            random.shuffle(pool)
        for sc, nm, tags_list, _dsv in pool:

            if len(plan) >= need:
                break
            if already_in_plan(nm):
                continue
            if can_add(tags_list):
                add_recipe(nm, tags_list)

    # итоговый DataFrame
    return pd.DataFrame(plan, columns=["Recipe", "Tags", "Servings"])


def aggregate_shopping(df_plan, df_ing, cfg, df_pantry, df_recipes):
    servings = int(cfg["ServingsPerMeal"])

    # ----- карты справочников ----
    # 1) рецепт -> DefaultServings (из Recipes)
    default_map = dict(
        zip(
            df_recipes["Recipe"].astype(str).str.strip().str.lower(),
            pd.to_numeric(df_recipes["DefaultServings"], errors="coerce")
            .fillna(servings)
            .astype(float),
        )
    )

    # 2) рецепт -> сколько готовим на эту неделю (из плана)
    serv_map = dict(
        zip(
            df_plan["Recipe"].astype(str).str.strip().str.lower(),
            pd.to_numeric(df_plan["Servings"], errors="coerce")
            .fillna(servings)
            .astype(float),
        )
    )

    # -----локальный парсер количества ----
    import re

    def parse_qty(value):
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().lower().replace(",", ".")
        frac = {
            "1/2": "0.5",
            "1/4": "0.25",
            "3/4": "0.75",
            "1/3": "0.3333",
            "2/3": "0.6667",
            "1/8": "0.125",
            "3/8": "0.375",
            "5/8": "0.625",
            "7/8": "0.875",
        }
        for k, v in frac.items():
            s = s.replace(k, v)
        s = (
            s.replace(" или ", "-")
            .replace("-", "-")
            .replace("-", "-")
            .replace(" to ", "-")
        )
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        return sum(map(float, nums)) / len(nums) if nums else 0.0

    rows = []

    # Для ускорения подготовим нижний регистр Recipe в Ingredients
    df_ing_local = df_ing.copy()
    df_ing_local["__RecipeKey"] = (
        df_ing_local["Recipe"].astype(str).str.strip().str.lower()
    )

    for _, plan_row in df_plan.iterrows():
        recipe_name = str(plan_row["Recipe"]).strip()
        recipe_key = recipe_name.lower()
        target_serv = float(serv_map.get(recipe_key, servings))

        # подтаблица ингридиентов для этого рецепта (без учета регистра)
        sub = df_ing_local[df_ing_local["__RecipeKey"] == recipe_key]
        if sub.empty:
            continue

        for _, ing in sub.iterrows():
            # 1) сырой объем из ячейки (это КОЛ-ВО на ВЕСЬ рецепт)
            qty_raw = parse_qty(ing["QtyPerServing"])

            # 2) масштабирование по порциям
            default_serv = float(default_map.get(recipe_key, max(1.0, target_serv)))
            qty = qty_raw * (target_serv / max(default_serv, 1.0))

            # 3) нормализация единиц
            unit = ing.get("Unit", "")
            base_unit, k = normalize_unit(unit)
            qty_base = qty * k

            # 4) опциональный перевод ложек в мл (если включен в config)
            """
            if cfg.get("ConvertSpoonsToMl", False) and base_unit in ("ст.л.", "ч.л."):
                ml = SPOON_TO_ML.get(base_unit, 1) * qty_base
                qty_base = ml
                base_unit = "мл"
            """

            # 5) источник покупки (рынок или магазин), если колонка есть
            source = ""
            if "Source" in ing.index and pd.notna(ing["Source"]):
                source = str(ing["Source"]).strip().lower()

            # 6) наименование товара
            item = str(ing["Item"]).strip().lower()

            rows.append(
                {"Item": item, "Qty": qty_base, "UnitBase": base_unit, "Source": source}
            )

    # ----- датафрейм и агрегация ----
    if not rows:
        return pd.DataFrame(columns=["Item", "Qty", "Unit", "Source"])

    df = pd.DataFrame(rows)

    # вычитание кладовой (если используем - то вставляем свою фугкцию, иначе пропускаем)
    # df = apply_pantry_rowwise(df, df_pantry, normilize_unit ) # если есть такая функция
    # или просто ничего не делаем, если Pantry пустой

    # агрегируем (с Source, если колонка существует)
    group_cols = (
        ["Item", "UnitBase", "Source"]
        if "Source" in df.columns
        else ["Item", "UnitBase"]
    )
    agg = df.groupby(group_cols, as_index=False)["Qty"].sum()

    # никаких округленй
    agg["Qty"] = agg["Qty"].astype(float)

    # финальные имена/порядок
    agg = agg.rename(columns={"UnitBase": "Unit"})
    # красивое отображение Source
    if "Source" in agg.columns:
        disp = {"рынок": "Рынок", "магазин": "Магазин"}
        agg["Source"] = (
            agg["Source"]
            .astype(str)
            .str.strip()
            .map(lambda s: disp.get(s.lower(), s.title()))
        )
        agg = agg.sort_values(["Source", "Item", "Unit"]).reset_index(drop=True)
    else:
        agg = agg.sort_values(["Item", "Unit"]).reset_index(drop=True)

    # порядок колонок
    desired = ["Item", "Qty", "Unit", "Source"]
    agg = agg[[c for c in desired if c in agg.columns]]

    return agg


def update_history(xlsx_path, week_start, df_plan):
    """Добавляет выбранные рецепты недели в лист History исходного шаблона
    week_start: date (понедельник), df_plan: DataFrame c колонками как минимум ["Recipe", "Servings"]
    """

    # читаем текущий History (если листа нет - создадим пустой)
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    if "History" in xl.sheet_names:
        df_hist = xl.parse("History")
    else:
        df_hist = pd.DataFrame(columns=["WeekStart", "Recipe", "Servings"])

    # нормализуем колонки
    df_hist.columns = [str(c).strip() for c in df_hist.columns]
    for need in ["WeekStart", "Recipe", "Servings"]:
        if need not in df_hist.columns:
            df_hist[need] = pd.Series(dtype="object")

    # готовим строки к добавлению
    stamp = week_start.strftime("%Y-%m-%d")
    add = df_plan[["Recipe", "Servings"]].copy()
    add["WeekStart"] = stamp
    add = add[["WeekStart", "Recipe", "Servings"]]

    # склеиваем и убираем явные дубликаты (WeekStart + Recipe)
    df_hist_new = pd.concat([df_hist, add], ignore_index=True)
    df_hist_new = df_hist_new.drop_duplicates(
        subset=["WeekStart", "Recipe"], keep="last"
    )

    # пишем обратно в этот же Exel (заменим лист History)
    with pd.ExcelWriter(
        xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df_hist_new.to_excel(writer, sheet_name="History", index=False)


def main(week_start=None, xlsx_path=TEMPLATE_PATH, out_dir=BASE_DIR):
    if week_start is None:
        today = dt.date.today()
        # неделя начинается с понедельника
        week_start = today - dt.timedelta(days=today.weekday())
    elif isinstance(week_start, str):
        week_start = dt.datetime.strptime(week_start, "%Y-%m-%d").date()

    df_recipes, df_ing, df_targets, cfg, df_history, df_pantry = load_data(xlsx_path)

    # формируем план
    df_plan = choose_week(df_recipes, df_targets, df_history, cfg, week_start)

    # обновить History в шаблоне (Excel должен быть закрыт)
    update_history(xlsx_path, week_start, df_plan)

    # считаем список покупок
    df_shop = aggregate_shopping(df_plan, df_ing, cfg, df_pantry, df_recipes)

    # оформление Source и порядок колонок
    if not df_shop.empty:
        if "Source" in df_shop.columns:
            source_display_map = {"рынок": "Рынок", "магазин": "Магазин"}
            df_shop["Source"] = (
                df_shop["Source"]
                .astype(str)
                .str.strip()
                .map(lambda s: source_display_map.get(s.lower(), s.title()))
            )

    desired_order = ["Item", "Qty", "Unit", "Source"]
    df_shop = df_shop[[col for col in desired_order if col in df_shop.columns]]

    if "Source" in df_shop.columns:
        order = ["Рынок", "Магазин"]
        df_shop["Source"] = pd.Categorical(
            df_shop["Source"], categories=order, ordered=True
        )
        df_shop = df_shop.sort_values(["Source", "Item"]).reset_index(drop=True)

    # Сохраняем один Excel с двумЯ листами
    stamp = week_start.strftime("%Y-%m-%d")
    out_xlsx = os.path.join(out_dir, f"weekly_plan_{stamp}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
        df_plan.to_excel(writer, sheet_name="Plan", index=False)
        df_shop.to_excel(writer, sheet_name="ShoppingList", index=False)

    print("Сохранено в:", out_xlsx)
    return out_xlsx, df_plan, df_shop


if __name__ == "__main__":
    out_xlsx, df_plan, df_shop = main()
    print("Сохранено в:", out_xlsx)
