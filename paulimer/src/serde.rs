use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use crate::clifford::CliffordUnitary;
use crate::clifford::CliffordUnitaryModPauli;
use crate::pauli::SparsePauli;
use crate::pauli::SparsePauliProjective;

impl Serialize for CliffordUnitary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{self:#}"))
    }
}

impl<'de> Deserialize<'de> for CliffordUnitary {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let string = String::deserialize(deserializer)?;
        string
            .parse()
            .map_err(|_| de::Error::custom("failed to parse CliffordUnitary"))
    }
}

impl Serialize for CliffordUnitaryModPauli {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{self:#}"))
    }
}

impl<'de> Deserialize<'de> for CliffordUnitaryModPauli {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let string = String::deserialize(deserializer)?;
        string
            .parse()
            .map_err(|_| de::Error::custom("failed to parse CliffordUnitaryModPauli"))
    }
}

impl Serialize for SparsePauli {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{self:#}"))
    }
}

impl<'de> Deserialize<'de> for SparsePauli {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let string = String::deserialize(deserializer)?;
        string
            .parse()
            .map_err(|_| de::Error::custom("failed to parse SparsePauli"))
    }
}

impl Serialize for SparsePauliProjective {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{self:#}"))
    }
}

impl<'de> Deserialize<'de> for SparsePauliProjective {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let string = String::deserialize(deserializer)?;
        string
            .parse()
            .map_err(|_| de::Error::custom("failed to parse SparsePauliProjective"))
    }
}
