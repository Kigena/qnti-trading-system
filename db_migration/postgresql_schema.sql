-- QNTI Trading System - PostgreSQL Schema Migration
-- Migrated from SQLite to PostgreSQL with optimizations
-- Date: 2025-01-08

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schema
CREATE SCHEMA IF NOT EXISTS qnti;

-- Set search path
SET search_path TO qnti, public;

-- ==========================================
-- PORTFOLIO MANAGER TABLES
-- ==========================================

-- Accounts table (from qnti_portfolio_manager.py)
CREATE TABLE IF NOT EXISTS accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    account_type VARCHAR(50) NOT NULL CHECK (account_type IN ('demo', 'live', 'prop_firm', 'personal')),
    broker VARCHAR(255) NOT NULL,
    login VARCHAR(255) NOT NULL,
    server VARCHAR(255) NOT NULL,
    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    balance DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    equity DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    margin DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    free_margin DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    margin_level DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    profit DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    leverage INTEGER NOT NULL DEFAULT 100,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Portfolios table (from qnti_portfolio_manager.py)
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    account_ids JSONB NOT NULL DEFAULT '[]',
    allocation_method VARCHAR(50) NOT NULL CHECK (allocation_method IN ('equal_weight', 'risk_parity', 'market_cap', 'custom')),
    allocation_weights JSONB NOT NULL DEFAULT '{}',
    total_value DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    total_profit DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    daily_pnl DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    status VARCHAR(50) NOT NULL CHECK (status IN ('active', 'paused', 'stopped', 'error')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    performance_metrics JSONB DEFAULT '{}'
);

-- Positions table (from qnti_portfolio_manager.py)
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    volume DECIMAL(10,2) NOT NULL,
    open_price DECIMAL(15,5) NOT NULL,
    current_price DECIMAL(15,5) NOT NULL,
    profit_loss DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    swap DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    commission DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    open_time TIMESTAMP WITH TIME ZONE NOT NULL,
    magic_number INTEGER,
    comment TEXT
);

-- Portfolio snapshots table (from qnti_portfolio_manager.py)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_value DECIMAL(15,2) NOT NULL,
    total_profit DECIMAL(15,2) NOT NULL,
    daily_pnl DECIMAL(15,2) NOT NULL,
    account_values JSONB NOT NULL DEFAULT '{}',
    positions_count INTEGER NOT NULL DEFAULT 0,
    metrics JSONB DEFAULT '{}'
);

-- ==========================================
-- TRADE MANAGER TABLES
-- ==========================================

-- Trades table (from qnti_core_system.py)
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id VARCHAR(255) NOT NULL UNIQUE,
    magic_number INTEGER,
    symbol VARCHAR(20) NOT NULL,
    trade_type VARCHAR(50) NOT NULL,
    lot_size DECIMAL(10,2) NOT NULL,
    open_price DECIMAL(15,5) NOT NULL,
    close_price DECIMAL(15,5),
    stop_loss DECIMAL(15,5),
    take_profit DECIMAL(15,5),
    open_time TIMESTAMP WITH TIME ZONE,
    close_time TIMESTAMP WITH TIME ZONE,
    profit DECIMAL(15,2),
    commission DECIMAL(15,2),
    swap DECIMAL(15,2),
    source VARCHAR(50) NOT NULL CHECK (source IN ('vision_ai', 'ea', 'manual', 'hybrid')),
    status VARCHAR(50) NOT NULL CHECK (status IN ('open', 'closed', 'pending', 'cancelled')),
    ea_name VARCHAR(255),
    ai_confidence DECIMAL(5,2),
    strategy_tags JSONB DEFAULT '[]',
    notes TEXT
);

-- EA Performance table (from qnti_core_system.py)
CREATE TABLE IF NOT EXISTS ea_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ea_name VARCHAR(255) NOT NULL UNIQUE,
    magic_number INTEGER,
    symbol VARCHAR(20),
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    total_profit DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    total_loss DECIMAL(15,2) NOT NULL DEFAULT 0.00,
    win_rate DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    profit_factor DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    max_drawdown DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    avg_trade_duration INTERVAL,
    last_trade_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL CHECK (status IN ('active', 'paused', 'blocked', 'error', 'stopped')),
    risk_score DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    confidence_level DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- EA PROFILING TABLES
-- ==========================================

-- EA Profiles table (from qnti_ea_profiling_system.py)
CREATE TABLE IF NOT EXISTS ea_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ea_name VARCHAR(255) NOT NULL UNIQUE,
    magic_number INTEGER,
    symbol VARCHAR(20),
    strategy_type VARCHAR(50) NOT NULL,
    description TEXT,
    profile_data JSONB DEFAULT '{}',
    confidence_score DECIMAL(5,2) NOT NULL DEFAULT 0.50,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- EA Indicators table (from qnti_ea_profiling_system.py)
CREATE TABLE IF NOT EXISTS ea_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ea_name VARCHAR(255) NOT NULL REFERENCES ea_profiles(ea_name) ON DELETE CASCADE,
    indicator_name VARCHAR(255) NOT NULL,
    indicator_type VARCHAR(50) NOT NULL,
    parameters JSONB DEFAULT '{}',
    timeframe VARCHAR(10) NOT NULL,
    weight DECIMAL(5,2) NOT NULL DEFAULT 1.00,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- EA Analysis table (from qnti_ea_profiling_system.py)
CREATE TABLE IF NOT EXISTS ea_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ea_name VARCHAR(255) NOT NULL REFERENCES ea_profiles(ea_name) ON DELETE CASCADE,
    analysis_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    market_condition VARCHAR(50) NOT NULL,
    performance_score DECIMAL(5,2) NOT NULL,
    recommendations JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- ADDITIONAL SYSTEM TABLES
-- ==========================================

-- System logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(20) NOT NULL,
    module VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    user_id UUID,
    session_id UUID
);

-- Configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(255) NOT NULL UNIQUE,
    value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL UNIQUE,
    user_id UUID,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT true
);

-- ==========================================
-- INDEXES FOR PERFORMANCE
-- ==========================================

-- Accounts indexes
CREATE INDEX IF NOT EXISTS idx_accounts_broker ON accounts(broker);
CREATE INDEX IF NOT EXISTS idx_accounts_type ON accounts(account_type);
CREATE INDEX IF NOT EXISTS idx_accounts_active ON accounts(is_active);
CREATE INDEX IF NOT EXISTS idx_accounts_created ON accounts(created_at);
CREATE INDEX IF NOT EXISTS idx_accounts_updated ON accounts(last_updated);

-- Portfolios indexes
CREATE INDEX IF NOT EXISTS idx_portfolios_status ON portfolios(status);
CREATE INDEX IF NOT EXISTS idx_portfolios_created ON portfolios(created_at);
CREATE INDEX IF NOT EXISTS idx_portfolios_updated ON portfolios(last_updated);
CREATE INDEX IF NOT EXISTS idx_portfolios_account_ids ON portfolios USING gin(account_ids);

-- Positions indexes
CREATE INDEX IF NOT EXISTS idx_positions_account_id ON positions(account_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_side ON positions(side);
CREATE INDEX IF NOT EXISTS idx_positions_open_time ON positions(open_time);
CREATE INDEX IF NOT EXISTS idx_positions_magic_number ON positions(magic_number);
CREATE INDEX IF NOT EXISTS idx_positions_composite ON positions(account_id, symbol, open_time);

-- Portfolio snapshots indexes
CREATE INDEX IF NOT EXISTS idx_snapshots_portfolio_id ON portfolio_snapshots(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON portfolio_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_composite ON portfolio_snapshots(portfolio_id, timestamp);

-- Trades indexes
CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_source ON trades(source);
CREATE INDEX IF NOT EXISTS idx_trades_ea_name ON trades(ea_name);
CREATE INDEX IF NOT EXISTS idx_trades_magic_number ON trades(magic_number);
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time);
CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time);
CREATE INDEX IF NOT EXISTS idx_trades_composite ON trades(symbol, status, open_time);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_tags ON trades USING gin(strategy_tags);

-- EA Performance indexes
CREATE INDEX IF NOT EXISTS idx_ea_performance_name ON ea_performance(ea_name);
CREATE INDEX IF NOT EXISTS idx_ea_performance_magic ON ea_performance(magic_number);
CREATE INDEX IF NOT EXISTS idx_ea_performance_symbol ON ea_performance(symbol);
CREATE INDEX IF NOT EXISTS idx_ea_performance_status ON ea_performance(status);
CREATE INDEX IF NOT EXISTS idx_ea_performance_updated ON ea_performance(last_updated);

-- EA Profiles indexes
CREATE INDEX IF NOT EXISTS idx_ea_profiles_name ON ea_profiles(ea_name);
CREATE INDEX IF NOT EXISTS idx_ea_profiles_magic ON ea_profiles(magic_number);
CREATE INDEX IF NOT EXISTS idx_ea_profiles_strategy_type ON ea_profiles(strategy_type);
CREATE INDEX IF NOT EXISTS idx_ea_profiles_confidence ON ea_profiles(confidence_score);

-- EA Indicators indexes
CREATE INDEX IF NOT EXISTS idx_ea_indicators_name ON ea_indicators(ea_name);
CREATE INDEX IF NOT EXISTS idx_ea_indicators_type ON ea_indicators(indicator_type);
CREATE INDEX IF NOT EXISTS idx_ea_indicators_timeframe ON ea_indicators(timeframe);

-- EA Analysis indexes
CREATE INDEX IF NOT EXISTS idx_ea_analysis_name ON ea_analysis(ea_name);
CREATE INDEX IF NOT EXISTS idx_ea_analysis_date ON ea_analysis(analysis_date);
CREATE INDEX IF NOT EXISTS idx_ea_analysis_condition ON ea_analysis(market_condition);

-- System logs indexes
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_system_logs_module ON system_logs(module);
CREATE INDEX IF NOT EXISTS idx_system_logs_composite ON system_logs(timestamp, level, module);

-- System config indexes
CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(key);

-- User sessions indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_activity ON user_sessions(last_activity);

-- ==========================================
-- FUNCTIONS AND TRIGGERS
-- ==========================================

-- Function to update last_updated timestamp
CREATE OR REPLACE FUNCTION update_last_updated_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_accounts_last_updated BEFORE UPDATE ON accounts
    FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_portfolios_last_updated BEFORE UPDATE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_ea_performance_last_updated BEFORE UPDATE ON ea_performance
    FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_ea_profiles_last_updated BEFORE UPDATE ON ea_profiles
    FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();

CREATE TRIGGER update_system_config_last_updated BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();

-- Function to calculate portfolio metrics
CREATE OR REPLACE FUNCTION calculate_portfolio_total_value(portfolio_uuid UUID)
RETURNS DECIMAL(15,2) AS $$
DECLARE
    total_value DECIMAL(15,2) := 0;
    account_uuid UUID;
    weight DECIMAL(10,2);
BEGIN
    -- Get account IDs and weights from portfolio
    FOR account_uuid IN 
        SELECT jsonb_array_elements_text(account_ids)::UUID 
        FROM portfolios 
        WHERE id = portfolio_uuid
    LOOP
        -- Get weight for this account
        SELECT COALESCE((allocation_weights->>account_uuid::text)::DECIMAL(10,2), 0)
        INTO weight
        FROM portfolios 
        WHERE id = portfolio_uuid;
        
        -- Add weighted equity to total
        SELECT total_value + COALESCE(equity * weight, 0)
        INTO total_value
        FROM accounts 
        WHERE id = account_uuid;
    END LOOP;
    
    RETURN COALESCE(total_value, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to clean old snapshots (keep last 1000 per portfolio)
CREATE OR REPLACE FUNCTION cleanup_old_snapshots()
RETURNS void AS $$
BEGIN
    DELETE FROM portfolio_snapshots 
    WHERE id IN (
        SELECT id FROM (
            SELECT id,
                   ROW_NUMBER() OVER (PARTITION BY portfolio_id ORDER BY timestamp DESC) as rn
            FROM portfolio_snapshots
        ) ranked
        WHERE rn > 1000
    );
END;
$$ LANGUAGE plpgsql;

-- Function to calculate EA win rate
CREATE OR REPLACE FUNCTION calculate_ea_win_rate(ea_name_param VARCHAR(255))
RETURNS DECIMAL(5,2) AS $$
DECLARE
    total_trades INTEGER;
    winning_trades INTEGER;
    win_rate DECIMAL(5,2);
BEGIN
    SELECT COUNT(*) INTO total_trades
    FROM trades 
    WHERE ea_name = ea_name_param AND status = 'closed' AND profit IS NOT NULL;
    
    SELECT COUNT(*) INTO winning_trades
    FROM trades 
    WHERE ea_name = ea_name_param AND status = 'closed' AND profit > 0;
    
    IF total_trades > 0 THEN
        win_rate := (winning_trades::DECIMAL / total_trades::DECIMAL) * 100;
    ELSE
        win_rate := 0;
    END IF;
    
    RETURN win_rate;
END;
$$ LANGUAGE plpgsql;

-- ==========================================
-- VIEWS FOR COMMON QUERIES
-- ==========================================

-- View for active portfolios with current metrics
CREATE OR REPLACE VIEW active_portfolios_view AS
SELECT 
    p.id,
    p.name,
    p.description,
    p.total_value,
    p.total_profit,
    p.daily_pnl,
    p.status,
    p.created_at,
    p.last_updated,
    jsonb_array_length(p.account_ids) as account_count,
    calculate_portfolio_total_value(p.id) as calculated_total_value
FROM portfolios p
WHERE p.status = 'active';

-- View for EA performance summary
CREATE OR REPLACE VIEW ea_performance_summary AS
SELECT 
    ep.ea_name,
    ep.magic_number,
    ep.symbol,
    ep.total_trades,
    ep.winning_trades,
    ep.losing_trades,
    ep.win_rate,
    ep.total_profit,
    ep.total_loss,
    ep.profit_factor,
    ep.max_drawdown,
    ep.risk_score,
    ep.status,
    ep.last_trade_time,
    COUNT(t.id) as recent_trades_count
FROM ea_performance ep
LEFT JOIN trades t ON ep.ea_name = t.ea_name AND t.open_time >= (CURRENT_TIMESTAMP - INTERVAL '7 days')
GROUP BY ep.ea_name, ep.magic_number, ep.symbol, ep.total_trades, ep.winning_trades, 
         ep.losing_trades, ep.win_rate, ep.total_profit, ep.total_loss, ep.profit_factor, 
         ep.max_drawdown, ep.risk_score, ep.status, ep.last_trade_time;

-- View for recent trades with EA info
CREATE OR REPLACE VIEW recent_trades_view AS
SELECT 
    t.id,
    t.trade_id,
    t.symbol,
    t.trade_type,
    t.lot_size,
    t.open_price,
    t.close_price,
    t.profit,
    t.status,
    t.source,
    t.ea_name,
    t.open_time,
    t.close_time,
    ep.strategy_type,
    ep.confidence_score as ea_confidence
FROM trades t
LEFT JOIN ea_profiles ep ON t.ea_name = ep.ea_name
WHERE t.open_time >= (CURRENT_TIMESTAMP - INTERVAL '30 days')
ORDER BY t.open_time DESC;

-- ==========================================
-- PARTITIONING FOR LARGE TABLES
-- ==========================================

-- Partition system_logs by month
CREATE TABLE IF NOT EXISTS system_logs_template (
    LIKE system_logs INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create partitions for current and next month
CREATE TABLE IF NOT EXISTS system_logs_current PARTITION OF system_logs_template
    FOR VALUES FROM (date_trunc('month', CURRENT_DATE)) 
    TO (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month');

CREATE TABLE IF NOT EXISTS system_logs_next PARTITION OF system_logs_template
    FOR VALUES FROM (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month') 
    TO (date_trunc('month', CURRENT_DATE) + INTERVAL '2 months');

-- ==========================================
-- INITIAL CONFIGURATION DATA
-- ==========================================

-- Insert default system configuration
INSERT INTO system_config (key, value, description) VALUES
('system.version', '"1.0.0"', 'QNTI System Version'),
('database.migration_version', '"1.0.0"', 'Database Migration Version'),
('portfolio.default_currency', '"USD"', 'Default portfolio currency'),
('portfolio.max_accounts', '100', 'Maximum accounts per portfolio'),
('trading.max_open_trades', '50', 'Maximum open trades per EA'),
('risk.max_drawdown_alert', '0.20', 'Maximum drawdown alert threshold'),
('performance.cache_duration', '30', 'Performance cache duration in seconds'),
('backup.retention_days', '90', 'Backup retention period in days')
ON CONFLICT (key) DO NOTHING;

-- ==========================================
-- SECURITY AND PERMISSIONS
-- ==========================================

-- Create roles for different access levels
CREATE ROLE qnti_admin;
CREATE ROLE qnti_trader;
CREATE ROLE qnti_readonly;

-- Grant permissions to admin role
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA qnti TO qnti_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA qnti TO qnti_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA qnti TO qnti_admin;

-- Grant permissions to trader role
GRANT SELECT, INSERT, UPDATE ON accounts, portfolios, positions, trades, ea_performance TO qnti_trader;
GRANT SELECT ON ea_profiles, ea_indicators, ea_analysis TO qnti_trader;
GRANT INSERT ON system_logs TO qnti_trader;

-- Grant permissions to readonly role
GRANT SELECT ON ALL TABLES IN SCHEMA qnti TO qnti_readonly;

-- Row Level Security (RLS) can be enabled for multi-tenant scenarios
-- ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

-- ==========================================
-- VACUUM AND ANALYZE
-- ==========================================

-- Analyze tables for query planner
ANALYZE accounts;
ANALYZE portfolios;
ANALYZE positions;
ANALYZE portfolio_snapshots;
ANALYZE trades;
ANALYZE ea_performance;
ANALYZE ea_profiles;
ANALYZE ea_indicators;
ANALYZE ea_analysis;
ANALYZE system_logs;
ANALYZE system_config;

-- Set up auto-vacuum for optimal performance
ALTER TABLE accounts SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE portfolios SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE positions SET (autovacuum_vacuum_scale_factor = 0.05);
ALTER TABLE trades SET (autovacuum_vacuum_scale_factor = 0.05);
ALTER TABLE portfolio_snapshots SET (autovacuum_vacuum_scale_factor = 0.02);
ALTER TABLE system_logs SET (autovacuum_vacuum_scale_factor = 0.01);

COMMIT;

-- Migration completed successfully
SELECT 'QNTI PostgreSQL Schema Migration Completed Successfully!' as status;
