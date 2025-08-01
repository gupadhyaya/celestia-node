package core

import (
	"fmt"
	"strconv"

	"github.com/celestiaorg/celestia-node/libs/utils"
)

const (
	DefaultPort = "9090"
)

var MetricsEnabled bool

// Config combines all configuration fields for managing the relationship with a Core node.
type Config struct {
	EndpointConfig
	// AdditionalCoreEndpoints is a list of additional Celestia-Core endpoints to be used for
	// transaction submission. Must be provided as `host:port` pairs.
	AdditionalCoreEndpoints []EndpointConfig
}

type EndpointConfig struct {
	IP   string
	Port string
	// TLSEnabled specifies whether the connection is secure or not.
	// PLEASE NOTE: it should be set to true in order to handle XTokenPath.
	TLSEnabled bool
	// XTokenPath specifies the path to the directory with JSON file containing the X-Token for gRPC authentication.
	// The JSON file should have a key-value pair where the key is "x-token" and the value is the authentication token.
	// If left empty, the client will not include the X-Token in its requests.
	XTokenPath string
}

// DefaultConfig returns default configuration for managing the
// node's connection to a Celestia-Core endpoint.
func DefaultConfig() Config {
	return Config{
		EndpointConfig: EndpointConfig{
			IP:   "",
			Port: DefaultPort,
		},
		AdditionalCoreEndpoints: make([]EndpointConfig, 0),
	}
}

// Validate performs basic validation of the config.
func (cfg *Config) Validate() error {
	if !cfg.IsEndpointConfigured() {
		return nil
	}

	if err := cfg.validate(); err != nil {
		return err
	}

	for _, additionalCfg := range cfg.AdditionalCoreEndpoints {
		if err := additionalCfg.validate(); err != nil {
			return fmt.Errorf("nodebuilder/core: invalid additional core endpoint configuration: %w", err)
		}
	}

	return nil
}

func (cfg *EndpointConfig) validate() error {
	if cfg.Port == "" {
		return fmt.Errorf("nodebuilder/core: grpc port is not set")
	}

	ip, err := utils.SanitizeAddr(cfg.IP)
	if err != nil {
		return err
	}
	cfg.IP = ip
	_, err = strconv.Atoi(cfg.Port)
	if err != nil {
		return fmt.Errorf("nodebuilder/core: invalid grpc port: %s", err.Error())
	}

	return nil
}

// IsEndpointConfigured returns whether a core endpoint has been set
// on the config (true if set).
func (cfg *Config) IsEndpointConfigured() bool {
	return cfg.IP != ""
}
