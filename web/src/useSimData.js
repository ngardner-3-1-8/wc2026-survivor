import { useState, useEffect } from 'react'

const BASE = import.meta.env.BASE_URL

async function fetchJSON(file) {
  const res = await fetch(`${BASE}data/${file}`)
  if (!res.ok) throw new Error(`Failed to load ${file}: ${res.status}`)
  return res.json()
}

export function useSimData() {
  const [data, setData]     = useState(null)
  const [error, setError]   = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetchJSON('meta.json'),
      fetchJSON('groups.json'),
      fetchJSON('knockout.json'),
      fetchJSON('survivor.json'),
    ])
      .then(([meta, groups, knockout, survivor]) => {
        setData({ meta, groups, knockout, survivor })
      })
      .catch(setError)
      .finally(() => setLoading(false))
  }, [])

  return { data, error, loading }
}
