# Load the data
data = pd.read_csv(r"E:\4F\Capstone\bonds_w_exp_returns (1).csv")

# Creating a DataFrame
df = pd.DataFrame(data)

# Set Date and SecurityId as index for grouping
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(['SecurityId', 'Date'], inplace=True)

cds_spread = 200  # Example CDS spread value

# Calculate CDS-Bond basis
df['cds_bond_basis'] = df['spread'] - cds_spread

# Set a threshold
threshold = 20

def calculate_trade_profit(row):
    if row['ExpectedReturn'] > 0:
        return row['ExpectedReturn'] + row['AccruedInterest'] - cds_spread / 365
    elif row['ExpectedReturn'] < 0:
        return -row['ExpectedReturn'] - row['AccruedInterest'] + cds_spread / 365
    else:
        return 0

df['trade_profit'] = df.apply(calculate_trade_profit, axis=1)

def basis_trading(row):
    if row['AccruedInterest'] >= row['trade_profit']:
        return 'Hold'
    elif row['cds_bond_basis'] > threshold and row['AccruedInterest'] < row['trade_profit']:
    # Short the bond, Long the CDS (Bond overvalued relative to CDS)
        return 'Buy Bond, Sell CDS'
    elif row['cds_bond_basis'] > -threshold and row['AccruedInterest'] < row['trade_profit']:
    # Long the bond, short the CDS (CDS overvalued relative to bond)
        return 'Sell Bond, Buy CDS'
    else:
        return 'Hold'

df['trade_signal'] = df.apply(basis_trading, axis=1)

df.reset_index(inplace=True)

df_filtered = df[df['trade_signal'] != 'Hold']

print(df_filtered[['SecurityId', 'Date','trade_signal', 'trade_profit']])
