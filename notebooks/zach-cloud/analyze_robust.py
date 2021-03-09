#%% 
import pandas as pd
import seaborn as sns
# %%

df_40 = pd.read_csv("/home/ubuntu/ContactTracing/output/run_VDGwz/results.csv", na_values=["None"])

df = pd.read_csv("/home/ubuntu/ContactTracing/output/run_GuCwm/results.csv", na_values=["None"])
#%%
df[df["method"] == "robust"]
df_robust = pd.concat([df, df_40])
df_robust.describe()
#%%
df_base = pd.read_csv("/home/ubuntu/ContactTracing/output/run_47Hyo/results.csv", na_values=["None"])

df_base.describe()
#%%
df = pd.concat([df_robust, df_base])
#%%
sns.barplot(x="budget", y="infected_v2", data=df, hue="method")

# %%
sns.barplot(x="budget", y="infected_v2", data=df, hue="compliance_rate")



#%%
g = sns.catplot(x="budget", y="infected_v2",
                hue="compliance_rate", col="from_cache",
                data=df, kind="bar",
                height=4, aspect=.7, orient="v")
#%%
g = sns.catplot(x="budget", y="infected_v2",
                hue="method", col="from_cache",
                data=df, kind="bar",
                height=4, aspect=.7, orient="v")


# %%
g.savefig("images/compare.png")
# %%
