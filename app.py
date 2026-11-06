import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# путь от файла, а не от текущей папки: так streamlit run работает
# из любого места
DATA_DIR = Path(__file__).parent / "data"

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


# 1. Загрузка данных

@st.cache_data
def load_data():
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")
    return orders, users, items


# 2. Объединение таблиц

def merge_data(orders, users, items):
    """Заказ + пользователь + товар в одной таблице"""
    # left: заказ остаётся, даже если пользователя или товара нет в
    # справочнике, такие строки потом видны как пропуски
    df = orders.merge(users, on="user_id", how="left")
    df = df.merge(items, on="item_id", how="left")
    return df


# 3. Очистка и подготовка

def clean_data(df):
    df = df.copy()

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["registration_date"] = pd.to_datetime(df["registration_date"],
                                             errors="coerce")
    for col in ["quantity", "price_per_unit", "base_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # без даты, количества или цены заказ не посчитать, такие удаляем.
    # Пропуски в описании товара и клиента не мешают, их заполняем
    df = df.dropna(subset=["order_date", "quantity", "price_per_unit"])
    for col in ["item_name", "category", "user_name", "city",
                "user_segment"]:
        df[col] = df[col].fillna("Unknown")

    # выручка по цене продажи из заказа, а не по base_price из
    # справочника: base_price сильно выше (см. выводы)
    df["revenue"] = df["quantity"] * df["price_per_unit"]
    df["weekday"] = df["order_date"].dt.day_of_week
    df["day"] = df["order_date"].dt.date
    return df


orders, users, items = load_data()
raw_df = merge_data(orders, users, items)
missing_before = raw_df.isnull().sum()
df = clean_data(raw_df)

# проверки целостности: что должно быть верно для любых заказов
checks = pd.DataFrame({
    "проверка": ["order_id не повторяется",
                 "количество больше 0",
                 "цена больше 0",
                 "регистрация не позже заказа"],
    "нарушений": [df["order_id"].duplicated().sum(),
                  (df["quantity"] <= 0).sum(),
                  (df["price_per_unit"] <= 0).sum(),
                  (df["registration_date"] > df["order_date"]).sum()],
})

st.title("Анализ продаж интернет-магазина")
st.caption("Данные за 1-10 февраля 2024: orders.csv, users.csv, items.csv")


# Фильтры (боковая панель)

st.sidebar.header("Фильтры")
all_categories = sorted(df["category"].unique())
chosen_categories = st.sidebar.multiselect("Категории", all_categories,
                                           default=all_categories)
date_range = st.sidebar.date_input(
    "Период",
    value=(df["day"].min(), df["day"].max()),
    min_value=df["day"].min(),
    max_value=df["day"].max(),
)

# пока в календаре выбрана только первая дата, приходит одна дата,
# а не пара
if len(date_range) != 2:
    st.info("Выберите конец периода в календаре слева")
    st.stop()
start_date, end_date = date_range

filtered = df[df["category"].isin(chosen_categories)
              & (df["day"] >= start_date) & (df["day"] <= end_date)]
if filtered.empty:
    st.warning("Под выбранные фильтры нет ни одного заказа")
    st.stop()

st.sidebar.write("Заказов под фильтром:", len(filtered))

tab_raw, tab_kpi, tab_charts, tab_conclusions = st.tabs(
    ["Сырые данные", "Ключевые показатели", "Визуализация",
     "Аналитические выводы"])


# 4. Сырые данные

with tab_raw:
    st.subheader("Объединённая таблица")
    st.write("Заказы, к которым присоединены пользователь (по user_id) "
             "и товар (по item_id). Фильтры слева действуют на все "
             "вкладки, кроме выводов.")
    st.dataframe(filtered.drop(columns=["weekday", "day"]))

    st.subheader("Проверка данных")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Пропуски после объединения:")
        st.dataframe(missing_before.rename("пропусков"))
    with col2:
        st.write("Типы после очистки:")
        st.dataframe(df.dtypes.astype(str).rename("тип"))
    st.write(f"Строк в исходных таблицах: orders {len(orders)}, "
             f"users {len(users)}, items {len(items)}. После объединения "
             f"и очистки: {len(df)}, пропусков было "
             f"{missing_before.sum()}.")
    st.write("Логические проверки:")
    st.dataframe(checks, hide_index=True)


# 5. Ключевые показатели

total_orders = filtered["order_id"].nunique()
total_revenue = filtered["revenue"].sum()
unique_users = filtered["user_id"].nunique()
average_check = total_revenue / total_orders

with tab_kpi:
    col1, col2 = st.columns(2)
    col1.metric("Общее количество заказов", total_orders)
    col2.metric("Общая выручка, руб.", f"{total_revenue:.2f}")
    col1.metric("Уникальных пользователей", unique_users)
    col2.metric("Средний чек, руб.", f"{average_check:.2f}")
    st.caption("Средний чек = выручка / количество заказов. "
               "В каждом заказе один товар (возможно, в нескольких штуках).")

    st.subheader("Выручка по сегментам клиентов")
    segments = filtered.pivot_table(index="user_segment",
                                    values=["order_id", "revenue"],
                                    aggfunc={"order_id": "count",
                                             "revenue": "sum"})
    segments = segments.rename(columns={"order_id": "заказов",
                                        "revenue": "выручка"})
    segments["средний чек"] = (segments["выручка"]
                               / segments["заказов"]).round(2)
    st.dataframe(segments.sort_values("выручка", ascending=False))


# 6. Визуализация

with tab_charts:
    # топ-10 товаров: самый большой сверху, поэтому сортирую по
    # возрастанию (barh рисует первую строку снизу)
    st.subheader("Топ-10 товаров по выручке")
    top_items = (filtered.groupby("item_name")["revenue"].sum()
                 .sort_values(ascending=False).head(10)
                 .sort_values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_items.index, top_items.values, color="#4c72b0")
    ax.set_xlabel("Выручка, руб.")
    for i, value in enumerate(top_items.values):
        ax.text(value, i, f" {value:.0f}", va="center")
    ax.set_xlim(0, top_items.max() * 1.15)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Выручка по категориям")
    by_category = (filtered.groupby("category")["revenue"].sum()
                   .sort_values())
    share = by_category / by_category.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(by_category.index, by_category.values, color="#55a868")
    ax.set_xlabel("Выручка, руб.")
    for i, value in enumerate(by_category.values):
        ax.text(value, i, f" {share.iloc[i]:.1f}%", va="center")
    ax.set_xlim(0, by_category.max() * 1.15)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Заказы по дням недели")
    # 1 февраля 2024 четверг, поэтому Чт, Пт и Сб попали в период
    # дважды. Сравнивать дни можно только в пересчёте на один день
    orders_by_weekday = filtered.groupby("weekday")["order_id"].count()
    days_by_weekday = filtered.groupby("weekday")["day"].nunique()
    per_day = orders_by_weekday / days_by_weekday

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [WEEKDAYS[d] for d in orders_by_weekday.index]
    ax.bar(labels, orders_by_weekday.values, color="#c0c0c0",
           label="заказов всего")
    ax.plot(labels, per_day.values, color="#dd8452", marker="o",
            label="в среднем за один день")
    ax.set_ylabel("Заказов")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    counts = []
    for label, days in zip(labels, days_by_weekday.values):
        counts.append(f"{label} {days}")
    st.caption("Сколько раз день недели попал в период: "
               + ", ".join(counts))


# 7. Аналитические выводы

# выводы считаются по всем данным, без фильтров
category_revenue = (df.groupby("category")["revenue"].sum()
                    .sort_values(ascending=False))
top_category = category_revenue.index[0]
second_category = category_revenue.index[1]
top_category_share = category_revenue.iloc[0] / category_revenue.sum() * 100
category_gap = category_revenue.iloc[0] / category_revenue.iloc[1]

item_revenue = df.groupby("item_name")["revenue"].sum()
top_item = item_revenue.idxmax()
top_item_qty = df.loc[df["item_name"] == top_item, "quantity"].sum()
item_qty = df.groupby("item_name")["quantity"].sum()
qty_leaders = list(item_qty[item_qty == item_qty.max()].index)

orders_per_date = df.groupby("day")["order_id"].count()
days_per_weekday = df.groupby("weekday")["day"].nunique()
doubled_days = []
for weekday in days_per_weekday[days_per_weekday > 1].index:
    doubled_days.append(WEEKDAYS[weekday])

segment_check = df.groupby("user_segment")["revenue"].mean()

# сколько процентов от base_price составляет цена продажи
df["price_ratio"] = df["price_per_unit"] / df["base_price"]
cheapest = df.loc[df["price_ratio"].idxmin()]

with tab_conclusions:
    st.subheader("Выводы для руководства")
    st.caption("Посчитаны по всему периоду, фильтры на них не влияют.")

    st.markdown(f"""
1. Основная выручка приходится на категорию "{top_category}":
{top_category_share:.1f}% всей выручки. Следующая за ней категория
"{second_category}" даёт в {category_gap:.1f} раза меньше. Выручка держится
на одной категории, и просадка в ней сразу ударит по всему магазину.

2. Лидер по выручке: {top_item} ({item_revenue.max():.2f} руб.,
{top_item_qty:.0f} шт.). По количеству штук лидеры другие:
{', '.join(qty_leaders)} (по {item_qty.max()} шт.). То есть больше всего
денег дают не самые массовые, а дорогие товары.

3. Пика по дням недели нет. На графике больше всего заказов в
{', '.join(doubled_days)}, но только потому, что эти дни попали в период
дважды. В пересчёте на один день заказов почти всегда
{orders_per_date.median():.0f} (от {orders_per_date.min()} до
{orders_per_date.max()}). Делать акции под конкретный день недели по этим
данным оснований нет.

4. Самый высокий средний чек у новых клиентов:
{segment_check['new']:.2f} руб. против {segment_check['regular']:.2f}
у постоянных и {segment_check['vip']:.2f} у VIP. Получается, VIP-клиенты
покупают дешевле всех. Стоит проверить, по каким правилам присваивается
сегмент VIP.

5. Цены в заказах не сходятся со справочником. Цена продажи в среднем
(медиана) всего {df['price_ratio'].median() * 100:.0f}% от base_price из
items.csv. Сильнее всего расходится "{cheapest['item_name']}": цена
продажи {cheapest['price_per_unit']:.2f} при базовой
{cheapest['base_price']:.2f}. Выручка посчитана по цене из заказов. Перед
тем как принимать решения по дашборду, расхождение нужно выяснить у
владельцев данных.

Ограничение: в данных всего {len(df)} заказов за {df['day'].nunique()}
дней, поэтому выводы предварительные, их стоит перепроверить на более
длинном периоде.
""")
