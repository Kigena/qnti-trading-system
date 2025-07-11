#!/bin/bash

# PostgreSQL Database Initialization Script
# This script runs after the schema has been created

set -e

# Function to log messages
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Database connection parameters
DB_NAME=${POSTGRES_DB:-qnti_trading}
DB_USER=${POSTGRES_USER:-qnti_user}

log_info "Starting database initialization for QNTI Trading System"

# Connect to the database and run initialization
psql -v ON_ERROR_STOP=1 --username "$DB_USER" --dbname "$DB_NAME" <<-EOSQL
    -- Set search path
    SET search_path TO qnti, public;
    
    -- Create additional roles if they don't exist
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'qnti_admin') THEN
            CREATE ROLE qnti_admin;
            GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA qnti TO qnti_admin;
            GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA qnti TO qnti_admin;
            GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA qnti TO qnti_admin;
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'qnti_trader') THEN
            CREATE ROLE qnti_trader;
            GRANT SELECT, INSERT, UPDATE ON accounts, portfolios, positions, trades, ea_performance TO qnti_trader;
            GRANT SELECT ON ea_profiles, ea_indicators, ea_analysis TO qnti_trader;
            GRANT INSERT ON system_logs TO qnti_trader;
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'qnti_readonly') THEN
            CREATE ROLE qnti_readonly;
            GRANT SELECT ON ALL TABLES IN SCHEMA qnti TO qnti_readonly;
        END IF;
    END
    \$\$;
    
    -- Grant role to main user
    GRANT qnti_admin TO $DB_USER;
    
    -- Insert default configuration if not exists
    INSERT INTO system_config (key, value, description) VALUES
    ('system.initialized', 'true', 'System initialization flag'),
    ('system.init_timestamp', to_jsonb(now()::text), 'System initialization timestamp'),
    ('database.version', '"1.0.0"', 'Database schema version'),
    ('container.deployment', 'true', 'Container deployment flag')
    ON CONFLICT (key) DO NOTHING;
    
    -- Create default portfolio if not exists
    INSERT INTO portfolios (name, description, allocation_method, status) VALUES
    ('Default Portfolio', 'Default trading portfolio', 'equal_weight', 'active')
    ON CONFLICT DO NOTHING;
    
    -- Create sample EA profiles for testing
    INSERT INTO ea_profiles (ea_name, strategy_type, description, confidence_score) VALUES
    ('Sample_EA_1', 'trend_following', 'Sample trend following EA', 0.75),
    ('Sample_EA_2', 'mean_reversion', 'Sample mean reversion EA', 0.80),
    ('Sample_EA_3', 'scalping', 'Sample scalping EA', 0.65)
    ON CONFLICT (ea_name) DO NOTHING;
    
    -- Create indexes for performance optimization
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_performance ON trades(symbol, status, open_time) WHERE status = 'open';
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_performance ON positions(account_id, symbol) WHERE volume > 0;
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_logs_recent ON system_logs(timestamp) WHERE timestamp > (now() - interval '7 days');
    
    -- Update statistics
    ANALYZE accounts;
    ANALYZE portfolios;
    ANALYZE positions;
    ANALYZE trades;
    ANALYZE ea_performance;
    ANALYZE ea_profiles;
    ANALYZE system_logs;
    
    SELECT 'Database initialization completed successfully!' as status;
EOSQL

log_info "Database initialization completed successfully"

# Create a health check function
psql -v ON_ERROR_STOP=1 --username "$DB_USER" --dbname "$DB_NAME" <<-EOSQL
    -- Create health check function
    CREATE OR REPLACE FUNCTION qnti_health_check()
    RETURNS TABLE(
        component text,
        status text,
        details text
    ) AS \$\$
    BEGIN
        -- Check database connection
        RETURN QUERY SELECT 'database'::text, 'healthy'::text, 'Database connection OK'::text;
        
        -- Check schema
        IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'qnti') THEN
            RETURN QUERY SELECT 'schema'::text, 'healthy'::text, 'Schema exists'::text;
        ELSE
            RETURN QUERY SELECT 'schema'::text, 'unhealthy'::text, 'Schema missing'::text;
        END IF;
        
        -- Check tables
        IF (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'qnti') > 10 THEN
            RETURN QUERY SELECT 'tables'::text, 'healthy'::text, 'All tables exist'::text;
        ELSE
            RETURN QUERY SELECT 'tables'::text, 'unhealthy'::text, 'Missing tables'::text;
        END IF;
        
        -- Check system config
        IF EXISTS (SELECT 1 FROM qnti.system_config WHERE key = 'system.initialized') THEN
            RETURN QUERY SELECT 'config'::text, 'healthy'::text, 'System config OK'::text;
        ELSE
            RETURN QUERY SELECT 'config'::text, 'unhealthy'::text, 'System config missing'::text;
        END IF;
        
        RETURN;
    END;
    \$\$ LANGUAGE plpgsql;
    
    -- Grant execute permission
    GRANT EXECUTE ON FUNCTION qnti_health_check() TO qnti_trader, qnti_readonly;
EOSQL

log_info "Health check function created successfully"